#ifndef BAL_PROBLEM_H_
#define BAL_PROBLEM_H_
#include <Eigen/Dense>
#include <unordered_map>

typedef Eigen::Vector3d Landmark;
typedef Eigen::Vector2d Observation;
typedef std::pair<size_t, size_t> IndexPair;

namespace std {
    template<>
    struct hash<IndexPair> {
        bool operator()(IndexPair const& rhs) const noexcept {
            std::size_t h1 = std::hash<size_t>{}(rhs.first);
            std::size_t h2 = std::hash<size_t>{}(rhs.second);
            return h1 & (h2 << 1);
        }
    };
}
struct CameraParam {
    // R, t, f, k1, k2
    double params[9];
};

struct Problem {
    std::unordered_map<size_t, CameraParam> cameras_;
    std::unordered_map<size_t, Landmark> points_;
    std::unordered_map<IndexPair, Observation> observations_;
};


#endif  // BAL_PROBLEM_H_