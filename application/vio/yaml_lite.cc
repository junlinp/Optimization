#include "yaml_lite.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vio {
namespace {

std::string Trim(const std::string& s) {
  size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

// Strips a trailing "#..." comment that starts outside of any '[' ... ']'
// span on this (already list-free) line.
std::string StripComment(const std::string& s) {
  size_t hash = s.find('#');
  if (hash == std::string::npos) return s;
  return Trim(s.substr(0, hash));
}

int LeadingSpaces(const std::string& raw_line) {
  int n = 0;
  while (n < static_cast<int>(raw_line.size()) && raw_line[n] == ' ') ++n;
  return n;
}

double ParseDoubleOrThrow(const std::string& token, const std::string& context) {
  const std::string t = Trim(token);
  if (t.empty()) {
    throw std::runtime_error("yaml_lite: empty numeric token in " + context);
  }
  char* end = nullptr;
  double value = std::strtod(t.c_str(), &end);
  if (end != t.c_str() + t.size()) {
    throw std::runtime_error("yaml_lite: malformed number '" + t + "' in " + context);
  }
  return value;
}

std::vector<double> ParseNumberList(const std::string& inside_brackets,
                                    const std::string& context) {
  std::vector<double> values;
  std::istringstream iss(inside_brackets);
  std::string token;
  while (std::getline(iss, token, ',')) {
    const std::string t = Trim(token);
    if (t.empty()) continue;
    values.push_back(ParseDoubleOrThrow(t, context));
  }
  return values;
}

}  // namespace

YamlLite ParseYamlLite(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("yaml_lite: could not open file: " + path);
  }

  YamlLite result;
  std::string current_top_key;

  bool accumulating = false;
  std::string accumulate_key;
  std::string accumulate_buffer;

  std::string raw_line;
  while (std::getline(ifs, raw_line)) {
    if (accumulating) {
      if (!accumulate_buffer.empty()) accumulate_buffer += ' ';
      accumulate_buffer += Trim(raw_line);
      size_t close = accumulate_buffer.find(']');
      if (close == std::string::npos) continue;
      size_t open = accumulate_buffer.find('[');
      if (open == std::string::npos || open > close) {
        throw std::runtime_error("yaml_lite: malformed list for key '" +
                                 accumulate_key + "' in " + path);
      }
      result.number_lists[accumulate_key] = ParseNumberList(
          accumulate_buffer.substr(open + 1, close - open - 1), accumulate_key);
      accumulating = false;
      accumulate_key.clear();
      accumulate_buffer.clear();
      continue;
    }

    const std::string trimmed = Trim(raw_line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    const int indent = LeadingSpaces(raw_line);
    const size_t colon = trimmed.find(':');
    if (colon == std::string::npos) continue;  // not a key: value line, ignore

    const std::string key_part = Trim(trimmed.substr(0, colon));
    const std::string rest = Trim(trimmed.substr(colon + 1));

    std::string full_key;
    if (indent == 0) {
      current_top_key = key_part;
      full_key = key_part;
    } else {
      full_key = current_top_key.empty() ? key_part : current_top_key + "." + key_part;
    }

    if (rest.empty()) {
      continue;  // block header, e.g. "T_BS:" -- value(s) follow on nested lines
    }

    if (rest[0] == '[') {
      size_t close = rest.find(']');
      if (close != std::string::npos) {
        result.number_lists[full_key] =
            ParseNumberList(rest.substr(1, close - 1), full_key);
      } else {
        accumulating = true;
        accumulate_key = full_key;
        accumulate_buffer = rest;
      }
      continue;
    }

    result.scalars[full_key] = StripComment(rest);
  }

  if (accumulating) {
    throw std::runtime_error("yaml_lite: unterminated list for key '" +
                             accumulate_key + "' in " + path);
  }

  return result;
}

}  // namespace vio
