#pragma once

#include "cnn/cnn.h"
#include "cnn/nodes.h"

namespace cnn { namespace math {

    int levenshtein_distance(const std::vector<std::string> &s1, const std::vector<std::string> &s2);

} }

