#ifndef VIO_YAML_LITE_H_
#define VIO_YAML_LITE_H_
#include <map>
#include <string>
#include <vector>

namespace vio {

// Handles exactly: '#' comments, blank lines, flat "key: value" scalars,
// flat "key: [a, b, c, ...]" inline numeric lists (which may span multiple
// lines, as the real T_BS.data blocks do), and one level of 2-space-indented
// nesting collapsed into a dotted key, e.g.
//   T_BS:
//     cols: 4
//     rows: 4
//     data: [1.0, 0.0, ...]
// yields scalars["T_BS.cols"]="4", scalars["T_BS.rows"]="4",
// number_lists["T_BS.data"]={1.0, 0.0, ...}.
//
// A bare "T_BS: identity" (no nested block) is not special-cased: callers
// that need identity as a fallback check scalars["T_BS"] == "identity"
// themselves.
struct YamlLite {
  std::map<std::string, std::string> scalars;
  std::map<std::string, std::vector<double>> number_lists;
};

// Throws std::runtime_error if the file cannot be opened.
YamlLite ParseYamlLite(const std::string& path);

}  // namespace vio
#endif  // VIO_YAML_LITE_H_
