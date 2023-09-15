#include "daba_subproblem_manager.h"

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>

#include "bal/cost_function_auto.h"

namespace {
template <class T, int DIM>
void NesteorvStep(const T* previous, T* current, T* target,
                  double nesteorv_coeeficient) {
  for (int i = 0; i < DIM; i++) {
    target[i] = current[i] + nesteorv_coeeficient * (current[i] - previous[i]);
  }
}

class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()> > tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty()) return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
      }
    });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) worker.join();
}
}  // namespace

void DABASubProblemManager::Solve(Problem& problem) {
  for (const auto& [camera_index, camera_parameters] : problem.cameras_) {
    camera_parameters_[camera_index] = camera_parameters.array();
    condition_camera_parameters_[camera_index] = camera_parameters.array();
    previous_camera_parameters_[camera_index] = camera_parameters.array();
  }

  for (const auto& [point_index, point_parameters] : problem.points_) {
    point_parameters_[point_index] = point_parameters.array();
    condition_point_parameters_[point_index] = point_parameters.array();
    previous_point_parameters_[point_index] = point_parameters.array();
  }

  std::map<int64_t, ceres::Problem> camera_cost_functions;
  std::map<int64_t, ceres::Problem> point_cost_functions;

  for (auto [index_pair, uv] : problem.observations_) {
    int64_t camera_index = index_pair.first;
    int64_t landmark_index = index_pair.second;

    ceres::CostFunction* camera_costfunction =
        new ceres::AutoDiffCostFunction<CameraSurrogateCostFunction, 3, 9>(
            new CameraSurrogateCostFunction(
                condition_camera_parameters_[camera_index].data(),
                condition_point_parameters_[landmark_index].data(), uv.u(),
                uv.v()));
    ceres::CostFunction* point_costfunction =
        new ceres::AutoDiffCostFunction<LandmarkSurrogatecostFunction, 3, 3>(
            new LandmarkSurrogatecostFunction(
                condition_camera_parameters_[camera_index].data(),
                condition_point_parameters_[landmark_index].data(), uv.u(),
                uv.v()));

    camera_cost_functions[camera_index].AddResidualBlock(
        camera_costfunction, nullptr, camera_parameters_[camera_index].data());
    point_cost_functions[landmark_index].AddResidualBlock(
        point_costfunction, nullptr, point_parameters_[landmark_index].data());
  }

  int iteration = 0;
  int max_iteration = 256;
  std::cout << "Start loop" << std::endl;
  double last_error = std::numeric_limits<double>::max();
  double s = 1;

  while (iteration++ < max_iteration) {
    double error = 0.0;
    std::mutex error_mutex;
    double s_next = (std::sqrt(4 * s * s + 1) + 1) * 0.5;
    double nesteorv_coeeficient = (s - 1) / s_next;
    s = s_next;

    auto camera_functor = [&camera_cost_functions, &error,
                           &error_mutex](int64_t c_i) {
      ceres::Solver::Summary summary;
      ceres::Solver::Options options;
      options.max_num_iterations = 512;
      ceres::Solve(options, &(camera_cost_functions.at(c_i)), &summary);
      std::lock_guard<std::mutex> lk(error_mutex);
      error += summary.final_cost;
    };
    auto point_functor = [&point_cost_functions, &error,
                          &error_mutex](int64_t p_i) {
      ceres::Solver::Summary summary;
      ceres::Solver::Options options;
      options.max_num_iterations = 512;
      ceres::Solve(options, &(point_cost_functions.at(p_i)), &summary);
      std::lock_guard<std::mutex> lk(error_mutex);
      error += summary.final_cost;
    };

    {
      ThreadPool thread_pool(std::thread::hardware_concurrency());

      for (auto& [camera_index, problem] : camera_cost_functions) {
        thread_pool.enqueue(camera_functor, camera_index);
      }

      for (auto& [point_index, problem] : point_cost_functions) {
        thread_pool.enqueue(point_functor, point_index);
      }
    }

    std::cout << iteration << " final cost:" << error << std::endl;

    if (error <= last_error) {
      for (auto& [camera_index, p] : condition_camera_parameters_) {
        NesteorvStep<double, 9>(
            previous_camera_parameters_[camera_index].data(),
            camera_parameters_[camera_index].data(), p.data(),
            nesteorv_coeeficient);
        std::copy(camera_parameters_[camera_index].begin(),
                  camera_parameters_[camera_index].end(),
                  previous_camera_parameters_[camera_index].begin());
        std::copy(p.begin(), p.end(), camera_parameters_[camera_index].begin());
      }
      for (auto& [point_index, p] : condition_point_parameters_) {
        NesteorvStep<double, 3>(previous_point_parameters_[point_index].data(),
                                point_parameters_[point_index].data(), p.data(),
                                nesteorv_coeeficient);
        std::copy(point_parameters_[point_index].begin(),
                  point_parameters_[point_index].end(),
                  previous_point_parameters_[point_index].begin());
        std::copy(p.begin(), p.end(), point_parameters_[point_index].begin());
      }
      last_error = error;
    } else {
      for (auto& [camera_index, p] : previous_camera_parameters_) {
        std::copy(p.begin(), p.end(), camera_parameters_[camera_index].begin());
        std::copy(p.begin(), p.end(),
                  condition_camera_parameters_[camera_index].begin());
      }

      for (auto& [point_index, p] : previous_point_parameters_) {
        std::copy(p.begin(), p.end(), point_parameters_[point_index].begin());
        std::copy(p.begin(), p.end(),
                  condition_point_parameters_[point_index].begin());
      }
    }
  }

  for (auto pair : camera_parameters_) {
    std::copy(pair.second.begin(), pair.second.end(),
              problem.cameras_.at(pair.first).data());
  }
  for (auto pair : point_parameters_) {
    std::copy(pair.second.begin(), pair.second.end(),
              problem.points_.at(pair.first).data());
  }
}