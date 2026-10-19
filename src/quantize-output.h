#pragma once

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <system_error>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace clap::detail {
// Own an exclusively created file beside the destination. Publication is one
// no-replace hard-link operation, never an exists/rename pair. Filesystems that
// cannot create hard links fail safely instead of falling back to clobbering.
class QuantizeOutput {
public:
    explicit QuantizeOutput(const std::filesystem::path &destination) : destination_(destination) {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            temporary_ = destination;
            temporary_ += ".tmp-" + std::to_string(random()) + "-" + std::to_string(random());
            int fd = -1;
#ifdef _WIN32
            const int error = _wsopen_s(&fd, temporary_.c_str(),
                _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYRW, _S_IREAD | _S_IWRITE);
#else
            fd = ::open(temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
            const int error = fd < 0 ? errno : 0;
#endif
            if (fd < 0) {
                if (error == EEXIST) continue;
                throw std::system_error(error, std::generic_category(), "cannot create temporary output");
            }
#ifdef _WIN32
            stream_ = _fdopen(fd, "wb");
#else
            stream_ = ::fdopen(fd, "wb");
#endif
            if (!stream_) {
                const int error = errno;
#ifdef _WIN32
                _close(fd);
#else
                ::close(fd);
#endif
                std::error_code ignored;
                std::filesystem::remove(temporary_, ignored);
                throw std::system_error(error, std::generic_category(), "cannot open temporary stream");
            }
            return;
        }
        throw std::runtime_error("cannot create a unique temporary output");
    }
    QuantizeOutput(const QuantizeOutput &) = delete;
    QuantizeOutput &operator=(const QuantizeOutput &) = delete;
    ~QuantizeOutput() {
        if (stream_) std::fclose(stream_);
        std::error_code ignored;
        std::filesystem::remove(temporary_, ignored);
    }
    void write(const void *data, size_t size) {
        if (!stream_ || std::fwrite(data, 1, size, stream_) != size)
            throw std::runtime_error("cannot write output");
    }
    void close() {
        if (!stream_) throw std::runtime_error("output already closed");
        auto *stream = stream_;
        stream_ = nullptr;
        if (std::fclose(stream) != 0) throw std::runtime_error("cannot finish output");
    }
    void publish() {
        if (stream_) throw std::runtime_error("output must be closed before publication");
        std::filesystem::create_hard_link(temporary_, destination_);
        std::filesystem::remove(temporary_);
        temporary_.clear();
    }
private:
    std::filesystem::path destination_, temporary_;
    std::FILE *stream_ = nullptr;
};
} // namespace clap::detail
