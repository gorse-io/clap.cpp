#include <CLI/CLI.hpp>

#include <iostream>

int main(int argc, char **argv) {
    CLI::App app{"clap.cpp"};
    app.add_subcommand("serve", "Print hello world")->callback([]() {
        std::cout << "hello world\n";
    });

    CLI11_PARSE(app, argc, argv);

    if (app.get_subcommands().empty()) {
        std::cout << "clap.cpp\n";
    }
    return 0;
}
