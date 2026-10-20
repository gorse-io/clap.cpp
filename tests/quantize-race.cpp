// Linux-only deterministic regression seam: create a competitor immediately
// before the real publication syscall, after any userspace exists() check.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

static void compete(const char *destination) {
    const char *target = std::getenv("CLAP_QUANTIZE_RACE_DESTINATION");
    if (!target || std::strcmp(target, destination) != 0) return;
    const int fd = ::open(destination, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return;
    constexpr char content[] = "concurrent destination\n";
    const auto ignored = ::write(fd, content, sizeof(content) - 1);
    (void) ignored;
    ::close(fd);
}
extern "C" int rename(const char *source, const char *destination) noexcept {
    const auto real = reinterpret_cast<int (*)(const char *, const char *)>(dlsym(RTLD_NEXT, "rename"));
    if (!real) { errno = ENOSYS; return -1; }
    compete(destination);
    return real(source, destination);
}
extern "C" int link(const char *source, const char *destination) noexcept {
    const auto real = reinterpret_cast<int (*)(const char *, const char *)>(dlsym(RTLD_NEXT, "link"));
    if (!real) { errno = ENOSYS; return -1; }
    compete(destination);
    return real(source, destination);
}
