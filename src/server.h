#pragma once

#include <string>

int run_server(const std::string &host, int port, const std::string &model_path, int threads);
