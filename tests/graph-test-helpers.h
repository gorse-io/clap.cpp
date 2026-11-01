#pragma once
#include "graph.h"

inline clap::detail::HostMatrix graph_linear(const clap::detail::GraphRuntime &runtime,
                                             const clap::detail::HostMatrix &input,
                                             const std::string &name, bool bias = true) {
    clap::detail::Graph graph(runtime);
    auto *output = graph.linear(graph.input(input), name, bias);
    clap::detail::HostMatrix result(int(output->ne[1]), int(output->ne[0]));
    result.data = graph.run(output);
    return result;
}
