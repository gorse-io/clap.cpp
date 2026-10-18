#include <httplib.h>

int main() {
    httplib::Server server;
    server.Get("/", [](const httplib::Request &, httplib::Response &response) {
        response.set_content("hello world", "text/plain");
    });
    return server.is_valid() ? 0 : 1;
}
