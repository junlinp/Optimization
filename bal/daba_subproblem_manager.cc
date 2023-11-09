#include "daba_subproblem_manager.h"

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <chrono>
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
      -> std::future<std::result_of_t<F(Args...)>> {
    using return_type = std::result_of_t<F(Args...)>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      // don't allow enqueueing after stopping the pool
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
  }
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

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

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) worker.join();
}

class Profiler {
 public:
  void StartProfile(const std::string& sample_name) {
    if (sample_total_count_.find(sample_name) == sample_total_count_.end()) {
      sample_total_count_.insert({sample_name, 0});
      sample_total_time_.insert({sample_name, 0});
    }

    sample_start_time_[sample_name] = std::chrono::high_resolution_clock::now();
  }

  void EndProfile(const std::string& sample_name) {
    std::chrono::time_point end_time =
        std::chrono::high_resolution_clock::now();

    assert(end_time > sample_start_time_[sample_name]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - sample_start_time_[sample_name]);

    sample_total_time_[sample_name] += duration.count();
    sample_total_count_[sample_name]++;
  }

  ~Profiler() {
    for (auto [sample_name, time] : sample_total_time_) {
      std::cout << sample_name << " means:"
                << static_cast<double>(time) / sample_total_count_[sample_name]
                << " ms " << std::endl;
    }
  }

  std::map<std::string, decltype(std::chrono::high_resolution_clock::now())>
      sample_start_time_;
  std::map<std::string, int64_t> sample_total_time_;
  std::map<std::string, int64_t> sample_total_count_;
};

}  // namespace

void DABASubProblemManager::Solve(Problem& problem) {
  for (const auto& [camera_index, camera_parameters] : problem.cameras_) {
    camera_parameters_[camera_index] = camera_parameters.array();
    condition_camera_parameters_[camera_index] = camera_parameters.array();
    previous_camera_parameters_[camera_index] = camera_parameters.array();
    current_camera_parameters_[camera_index] = camera_parameters.array();
    auxiliary_camera_parameters_[camera_index] = camera_parameters.array();
  }

  for (const auto& [point_index, point_parameters] : problem.points_) {
    point_parameters_[point_index] = point_parameters.array();
    condition_point_parameters_[point_index] = point_parameters.array();
    previous_point_parameters_[point_index] = point_parameters.array();
    current_point_parameters_[point_index] = point_parameters.array();
    auxiliary_point_parameters_[point_index] = point_parameters.array();
  }

  std::map<int64_t, ceres::Problem> camera_cost_functions;
  std::map<int64_t, ceres::Problem> point_cost_functions;
  std::vector<std::function<double()>> ray_cost_functions;
  
  std::map<int64_t, GradientDescentManager<9>> camera_gradient_descent;
  std::map<int64_t, GradientDescentManager<3>> point_gradient_descent;

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
    camera_gradient_descent[camera_index].Append(camera_costfunction);
      
    point_cost_functions[landmark_index].AddResidualBlock(
        point_costfunction, nullptr, point_parameters_[landmark_index].data());
    point_gradient_descent[landmark_index].Append(point_costfunction);

    ray_cost_functions.push_back(
        [uv = uv, this, camera_index, landmark_index]() {
          RayCostFunction ray_cost_func(uv.u(), uv.v());
          return ray_cost_func.EvaluateCost(
              camera_parameters_[camera_index].data(),
              point_parameters_[landmark_index].data());
        });
  }

  int iteration = 0;
  int max_iteration = 256;
  double function_tolerance = 1e-6;
  double last_error = std::numeric_limits<double>::max();
  double t = 0;
  double c = 0.0;
  double delta = 1e-5;
  double q = 1.0;
  double eta = 0.8;
  {
    for (auto& functor : ray_cost_functions) {
      c += functor();
    }
  }
  Profiler profiler;
  while (iteration++ < max_iteration) {
    double function_error = 0.0;
    double t_next = (std::sqrt(4 * t * t + 1) + 1) * 0.5;
    if (iteration == 1) {
        t_next = 1.0;
    }

    profiler.StartProfile("Update y");
    for (auto& [camera_index, condition_parameters] :
         condition_camera_parameters_) {
      auto current_parameters = CameraParam::ConvertLieAlgrebaToRotationMatrix(
          current_camera_parameters_.at(camera_index));
      auto previous_parameters = CameraParam::ConvertLieAlgrebaToRotationMatrix(
          previous_camera_parameters_.at(camera_index));
      auto auxiliary_parameters =
          CameraParam::ConvertLieAlgrebaToRotationMatrix(
              auxiliary_camera_parameters_.at(camera_index));
      
      std::array<double, 15> condition_p;
      for (int i = 0; i < 15; i++) {
        condition_p[i] =
            current_parameters[i] +
            t / t_next * (auxiliary_parameters[i] - current_parameters[i]) +
            (t - 1) / t_next * (current_parameters[i] - previous_parameters[i]);
      }

      condition_parameters = CameraParam::Project(condition_p);
      camera_parameters_[camera_index] = condition_parameters;
    }

    for (auto& [point_index, condition_parameters] :
         condition_point_parameters_) {
      auto& current_parameters = current_point_parameters_.at(point_index);
      auto& previous_parameters = previous_point_parameters_.at(point_index);
      auto& auxiliary_parameters = auxiliary_point_parameters_.at(point_index);

      for (int i = 0; i < 3; i++) {
        condition_parameters[i] =
            current_parameters[i] +
            t / t_next * (auxiliary_parameters[i] - current_parameters[i]) +
            (t - 1) / t_next * (current_parameters[i] - previous_parameters[i]);
      }
      point_parameters_[point_index] = condition_parameters;
    }
    profiler.EndProfile("Update y");

    auto camera_functor = [&camera_cost_functions, &camera_gradient_descent, this](int64_t c_i) {
      // ceres::Solver::Summary summary;
      // ceres::Solver::Options options;
      // options.max_num_iterations = 5;
      // ceres::Solve(options, &(camera_cost_functions.at(c_i)), &summary);

      auto& s = camera_gradient_descent.at(c_i);
      s.SetParameters(camera_parameters_.at(c_i).data());
      s.Step();
    };
    auto point_functor = [&point_cost_functions, &point_gradient_descent, this](int64_t p_i) {
      // ceres::Solver::Summary summary;
      // ceres::Solver::Options options;
      // options.max_num_iterations = 2;
      // ceres::Solve(options, &(point_cost_functions.at(p_i)), &summary);
      auto s = point_gradient_descent.at(p_i);
      s.SetParameters(point_parameters_.at(p_i).data());
      s.Step();
    };


    profiler.StartProfile("Solve z=f(y)");
    {
      ThreadPool thread_pool(std::thread::hardware_concurrency());

      for (auto& [camera_index, problem] : camera_cost_functions) {
        thread_pool.enqueue(camera_functor, camera_index);
      }

      for (auto& [point_index, problem] : point_cost_functions) {
        thread_pool.enqueue(point_functor, point_index);
      }
    }
    profiler.EndProfile("Solve z=f(y)");

    double auxiliary_error = 0.0;
    profiler.StartProfile("Compute f(z)");
    {
      for (auto& functor : ray_cost_functions) {
        auxiliary_error += functor();
      }
    }
    function_error = auxiliary_error;
    profiler.EndProfile("Compute f(z)");

    profiler.StartProfile("Compute norm(z - y)");
    double normal_diff = 0.0;
    for (auto& [camera_index, auxiliary_parameters] :
         auxiliary_camera_parameters_) {
      auxiliary_parameters = camera_parameters_[camera_index];
      const auto& condition_parameters =
          condition_camera_parameters_[camera_index];
      for (int i = 0; i < 9; i++) {
        normal_diff +=
            std::pow((auxiliary_parameters[i] - condition_parameters[i]), 2);
      }
    }

    for (auto& [point_index, auxiliary_parameters] :
         auxiliary_point_parameters_) {
      auxiliary_parameters = point_parameters_[point_index];
      const auto& condition_parameters =
          condition_point_parameters_[point_index];
      for (int i = 0; i < 3; i++) {
        normal_diff +=
            std::pow((auxiliary_parameters[i] - condition_parameters[i]), 2);
      }
    }
    profiler.EndProfile("Compute norm(z - y)");
    std::cout << iteration << " auxiliary_error : " << auxiliary_error
              << " c - delta * normal_diff : " << c - delta * normal_diff
              << std::endl;
    if (auxiliary_error <= c - delta * normal_diff) {
        profiler.StartProfile("x_next = z");
        for (auto& [camera_index, previous_parameters] : previous_camera_parameters_) {
            previous_parameters = current_camera_parameters_[camera_index];
            current_camera_parameters_[camera_index] = camera_parameters_[camera_index];
        }
        for (auto& [point_index, previous_parameters] : previous_point_parameters_) {
            previous_parameters = current_point_parameters_[point_index];
            current_point_parameters_[point_index] = point_parameters_[point_index];
        }
        profiler.EndProfile("x_next = z");
    } else {
        profiler.StartProfile("v = min f(x)");
      for (auto& [camera_index, current_parameters] :
           current_camera_parameters_) {
        condition_camera_parameters_[camera_index] = current_parameters;
        camera_parameters_[camera_index] = current_parameters;
      }
      for (auto& [point_index, current_parameters] :
           current_point_parameters_) {
        condition_point_parameters_[point_index] = current_parameters;
        point_parameters_[point_index] = current_parameters;
      }
      {
        ThreadPool thread_pool(std::thread::hardware_concurrency());

        for (auto& [camera_index, problem] : camera_cost_functions) {
          thread_pool.enqueue(camera_functor, camera_index);
        }

        for (auto& [point_index, problem] : point_cost_functions) {
          thread_pool.enqueue(point_functor, point_index);
        }
      }
      profiler.EndProfile("v = min f(x)");
      profiler.StartProfile("f(v)");
      double temp_error = 0.0;
      {
        for (auto& functor : ray_cost_functions) {
          temp_error += functor();
        }
      }
      profiler.EndProfile("f(v)");

      if (auxiliary_error <= temp_error) {
        for (auto& [camera_index, previous_parameters] : previous_camera_parameters_) {
            previous_parameters = current_camera_parameters_[camera_index];
            current_camera_parameters_[camera_index] = auxiliary_camera_parameters_[camera_index];
        }
        for (auto& [point_index, previous_parameters] : previous_point_parameters_) {
            previous_parameters = current_point_parameters_[point_index];
            current_point_parameters_[point_index] = auxiliary_point_parameters_[point_index];
        }
      } else {
        std::cout << iteration << " correct step is not a  good extrapolation" << std::endl;
        for (auto& [camera_index, previous_parameters] : previous_camera_parameters_) {
            previous_parameters = current_camera_parameters_[camera_index];
            current_camera_parameters_[camera_index] = camera_parameters_[camera_index];
            auxiliary_camera_parameters_[camera_index] = camera_parameters_[camera_index];
        }
        for (auto& [point_index, previous_parameters] : previous_point_parameters_) {
            previous_parameters = current_point_parameters_[point_index];
            current_point_parameters_[point_index] = point_parameters_[point_index];
            auxiliary_point_parameters_[point_index] = point_parameters_[point_index];
        }
        function_error = temp_error;
      }
    }
    t = t_next;
    double q_next = eta * q + 1;
    c = (eta * q * c + function_error) / q_next;
    q = q_next;
    std::cout << iteration << " function cost:" << function_error
              << std::endl;

    if (std::abs(function_error - last_error) / function_error < function_tolerance) {
      std::cout << "Function tolerance reached." << std::abs(function_error - last_error) / function_error << "quit early." << std::endl;
      break;
    }
    last_error = function_error;
  }

  for (auto pair : current_camera_parameters_) {
    std::copy(pair.second.begin(), pair.second.end(),
              problem.cameras_.at(pair.first).data());
  }
  for (auto pair : current_point_parameters_) {
    std::copy(pair.second.begin(), pair.second.end(),
              problem.points_.at(pair.first).data());
  }
}