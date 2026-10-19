#pragma once

#include <stdexcept>
#include <string>

namespace clap::detail {
inline void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("CLAP: " + message);
}
} // namespace clap::detail
