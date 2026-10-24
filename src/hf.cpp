#include "hf.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace clap {
namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
std::string env(const char *name) {
    const char *v = std::getenv(name);
    return v ? v : "";
}
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
std::string upper(std::string s) {
    for (char &c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}
bool hex(const std::string &s, size_t n) {
    return s.size() == n && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
void safe_name(const std::string &s) {
    const fs::path path(s);
    require(!s.empty() && !path.is_absolute() && !path.has_root_name() && !path.has_root_directory() &&
            s.front() != '/' && s.find_first_of("\\:<>\"|?*") == std::string::npos &&
            std::none_of(s.begin(), s.end(), [](unsigned char c) { return c < 32; }),
            "Unsafe repository filename");
    // Apply Windows rules on every platform: cache names must remain portable and
    // must not address drive-relative paths, alternate streams or device names.
    for (const auto &part : path) {
        const auto name = part.string();
        require(!name.empty() && name != "." && name != ".." && name.back() != '.' && name.back() != ' ',
                "Unsafe repository filename");
        const auto device = upper(name.substr(0, name.find('.')));
        require(device != "CON" && device != "PRN" && device != "AUX" && device != "NUL" &&
                device != "CONIN$" && device != "CONOUT$" &&
                !std::regex_match(device, std::regex(u8"(COM|LPT)([1-9]|\u00b9|\u00b2|\u00b3)")),
                "Unsafe repository filename");
    }
}
fs::path snapshot_path(const fs::path &root, const std::string &name) {
    safe_name(name);
    const auto base = fs::absolute(root).lexically_normal();
    const auto result = (base / name).lexically_normal();
    const auto prefix = std::mismatch(base.begin(), base.end(), result.begin(), result.end());
    require(prefix.first == base.end() && prefix.second != result.end(), "Unsafe snapshot path outside snapshot root");
    return result;
}
// Refuse symlinks in every directory component, including an explicitly supplied cache root.
void safe_path(const fs::path &p) {
    fs::path current;
    for (const auto &part : fs::absolute(p)) {
        current /= part;
        require(!fs::is_symlink(fs::symlink_status(current)), "Unsafe cache symlink: " + current.string());
    }
}
void directories(const fs::path &p) { safe_path(p); fs::create_directories(p); }
std::string read(const fs::path &p) {
    safe_path(p);
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
void publish_text(const fs::path &p, const std::string &text) {
    directories(p.parent_path());
    safe_path(p);
    const auto tmp = fs::path(p.string() + ".downloadInProgress");
    safe_path(tmp);
    { std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      out << text; out.close(); require(bool(out), "Cannot write cache metadata"); }
    fs::rename(tmp, p);
}
fs::path cache_root() {
    for (auto key : {"CLAP_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"})
        if (!env(key).empty()) return fs::absolute(env(key));
    if (!env("HF_HOME").empty()) return fs::absolute(fs::path(env("HF_HOME")) / "hub");
    fs::path base = env("XDG_CACHE_HOME");
    if (base.empty()) {
        const auto home = env("HOME").empty() ? env("USERPROFILE") : env("HOME");
        require(!home.empty(), "Set CLAP_CACHE, HOME or USERPROFILE to locate the model cache");
        base = fs::path(home) / ".cache";
    }
    return fs::absolute(base / "huggingface" / "hub");
}
std::string encode(const std::string &s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') out << c;
        else out << '%' << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    }
    return out.str();
}
struct Url { std::string origin, target; };
Url parse_url(const std::string &s) {
    std::smatch m;
    require(std::regex_match(s, m, std::regex("(https?://[^/?#@]+)(/[^#]*)?")), "Invalid HTTP endpoint or redirect URL");
    return {m[1], m[2].matched ? m[2].str() : "/"};
}
std::string next_url(const std::string &current, const std::string &location) {
    const auto u = parse_url(current);
    if (location.rfind("//", 0) == 0) return current.substr(0, current.find(':')) + ":" + location;
    if (location.rfind("/", 0) == 0) return u.origin + location;
    if (location.find("://") != std::string::npos) return location;
    return u.origin + u.target.substr(0, u.target.rfind('/') + 1) + location;
}
class Http {
    std::string endpoint_, token_;
public:
    Http(std::string endpoint, std::string token) : endpoint_(std::move(endpoint)), token_(std::move(token)) {}
    // Redirects are deliberately manual: credentials never leave the configured origin.
    httplib::Result get(std::string url, const httplib::Headers &extra = {},
                        const httplib::ResponseHandler &response = {},
                        const httplib::ContentReceiver &receiver = {}) {
        for (int redirect = 0; redirect < 10; ++redirect) {
            const auto u = parse_url(url);
            httplib::Client client(u.origin);
            // Native Windows/macOS trust stores do not use OpenSSL's environment defaults.
            // Honor an explicit CA bundle without disabling certificate or hostname checks.
            const auto ca_file = env("SSL_CERT_FILE");
            if (!ca_file.empty()) client.set_ca_cert_path(ca_file);
            client.set_connection_timeout(10);
            client.set_read_timeout(30);
            client.set_write_timeout(30);
            client.set_follow_location(false);
            client.set_decompress(false);
            httplib::Headers headers = extra;
            headers.emplace("Accept-Encoding", "identity");
            if (u.origin == parse_url(endpoint_).origin && !token_.empty())
                headers.emplace("Authorization", "Bearer " + token_);
            bool redirect_response = false;
            auto result = receiver ? client.Get(u.target, headers,
                [&](const httplib::Response &r) {
                    redirect_response = r.status >= 300 && r.status < 400;
                    return redirect_response || response(r);
                },
                [&](const char *data, size_t n) { return redirect_response || receiver(data, n); }) : client.Get(u.target, headers);
            if (!result || result->status < 300 || result->status >= 400) return result;
            const auto target = next_url(url, result->get_header_value("Location"));
            require(!(u.origin.rfind("https:", 0) == 0 && target.rfind("http:", 0) == 0), "Refusing HTTPS downgrade");
            url = target;
        }
        throw std::runtime_error("Too many HTTP redirects");
    }
    httplib::Result metadata(const std::string &url) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            auto r = get(url);
            if (r && r->status == 200) return r;
            if (r && r->status != 429 && r->status < 500)
                throw std::runtime_error("Hugging Face metadata HTTP " + std::to_string(r->status));
            if (attempt == 2) throw std::runtime_error("Hugging Face metadata request failed after 3 attempts");
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
        }
        throw std::runtime_error("Unreachable HTTP retry state");
    }
};
struct File { std::string name, oid; uint64_t size = 0; };
bool gguf(const std::string &s) { return s.size() > 5 && upper(s.substr(s.size() - 5)) == ".GGUF"; }
File select(std::vector<File> files, const std::string &filename, const std::string &quant) {
    std::sort(files.begin(), files.end(), [](const File &a, const File &b) { return a.name < b.name; });
    files.erase(std::remove_if(files.begin(), files.end(), [&](const File &f) {
        return !gguf(f.name) || (!filename.empty() && f.name != filename) ||
               (filename.empty() && !quant.empty() && upper(f.name).find(upper(quant)) == std::string::npos);
    }), files.end());
    require(!files.empty(), "No matching GGUF model found (safetensors are not converted)");
    if (filename.empty() && quant.empty()) {
        for (auto q : {"Q4_K_M", "Q8_0"}) {
            auto it = std::find_if(files.begin(), files.end(), [&](const File &f) { return upper(f.name).find(q) != std::string::npos; });
            if (it != files.end()) { std::rotate(files.begin(), it, it + 1); break; }
        }
    }
    require(!std::regex_search(files.front().name, std::regex("-[0-9]{5}-of-[0-9]{5}\\.gguf$", std::regex::icase)),
            "Split GGUF models are unsupported by the clap model loader; select an unsplit GGUF with -hff");
    return files.front();
}
bool cached(const fs::path &p, const fs::path &blobs, uint64_t size = 0) {
    safe_path(p.parent_path());
    if (fs::is_symlink(fs::symlink_status(p))) {
        const auto target = fs::weakly_canonical(p.parent_path() / fs::read_symlink(p));
        require(target.parent_path() == fs::weakly_canonical(blobs), "Unsafe snapshot symlink");
        safe_path(target);
    } else safe_path(p);
    return fs::is_regular_file(p) && (!size || fs::file_size(p) == size);
}
std::string sha256(const fs::path &p) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "Cannot initialize SHA256");
    std::ifstream in(p, std::ios::binary);
    char buffer[65536];
    while (in) { in.read(buffer, sizeof(buffer)); EVP_DigestUpdate(ctx.get(), buffer, static_cast<size_t>(in.gcount())); }
    require(in.eof(), "Cannot hash downloaded model");
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int n = 0;
    require(EVP_DigestFinal_ex(ctx.get(), digest, &n) == 1, "Cannot finish SHA256");
    std::ostringstream out;
    for (unsigned i = 0; i < n; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[i]);
    return out.str();
}
void download(Http &http, const std::string &url, const File &file, const fs::path &blob) {
    safe_path(blob);
    if (fs::is_regular_file(blob) && fs::file_size(blob) == file.size) return;
    const auto partial = fs::path(blob.string() + ".downloadInProgress");
    for (int attempt = 0; attempt < 3; ++attempt) {
        safe_path(partial);
        require(!fs::exists(partial) || (fs::is_regular_file(partial) && fs::hard_link_count(partial) == 1), "Unsafe partial download");
        uint64_t offset = fs::exists(partial) ? fs::file_size(partial) : 0;
        if (offset >= file.size) { fs::remove(partial); offset = 0; }
        std::ofstream out;
        uint64_t received = offset;
        bool accepted = false;
        httplib::Headers headers;
        if (offset) headers.emplace("Range", "bytes=" + std::to_string(offset) + "-");
        auto r = http.get(url, headers, [&](const httplib::Response &res) {
            if (res.status != 200 && res.status != 206) return false;
            if (res.status == 206) {
                const auto expected = "bytes " + std::to_string(offset) + "-" + std::to_string(file.size - 1) + "/" + std::to_string(file.size);
                if (res.get_header_value("Content-Range") != expected) return false;
            } else { offset = 0; received = 0; }
            if (res.has_header("Content-Length") && res.get_header_value("Content-Length") != std::to_string(file.size - offset)) return false;
            safe_path(partial);
            out.open(partial, std::ios::binary | (offset ? std::ios::app : std::ios::trunc));
            accepted = bool(out);
            return accepted;
        }, [&](const char *data, size_t n) {
            if (!accepted || n > file.size - received) return false;
            out.write(data, static_cast<std::streamsize>(n)); received += n;
            return bool(out);
        });
        if (out.is_open()) out.close();
        if (r && accepted && out && received == file.size) {
            if (hex(file.oid, 64) && sha256(partial) != file.oid) {
                fs::remove(partial);
                throw std::runtime_error("Downloaded model SHA256 mismatch");
            }
            safe_path(blob); fs::rename(partial, blob); return;
        }
        if (r && r->status >= 400 && r->status < 500 && r->status != 429)
            throw std::runtime_error("Model download HTTP " + std::to_string(r->status));
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
    }
    throw std::runtime_error("Model download failed after 3 attempts (HTTP, range or size validation); partial retained for resume");
}
} // namespace

std::string hf_download(const HfOptions &options) {
    std::string repo = options.repo, quant;
    const auto colon = repo.find(':');
    if (colon != std::string::npos) { quant = repo.substr(colon + 1); repo.resize(colon); require(!quant.empty(), "Empty HF quantization selector"); }
    require(std::regex_match(repo, std::regex("[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*")) && repo.find("..") == std::string::npos && repo.find("--") == std::string::npos,
            "Expected Hugging Face owner/repo[:quant]");
    if (!options.file.empty()) safe_name(options.file);
    const auto root = cache_root() / ("models--" + repo.substr(0, repo.find('/')) + "--" + repo.substr(repo.find('/') + 1));
    safe_path(root);
    const auto blobs = root / "blobs";
    safe_path(blobs);
    if (options.offline) {
        auto commit = read(root / "refs/main");
        while (!commit.empty() && std::isspace(static_cast<unsigned char>(commit.back()))) commit.pop_back();
        require(hex(commit, 40), "Offline cache miss: refs/main is missing or invalid");
        const auto snapshot = root / "snapshots" / commit;
        safe_path(snapshot);
        std::vector<File> files;
        if (fs::is_directory(snapshot)) {
            for (const auto &entry : fs::recursive_directory_iterator(snapshot)) {
                if (entry.is_directory()) { require(!entry.is_symlink(), "Unsafe cache directory symlink"); continue; }
                if (gguf(entry.path().string()) && cached(entry.path(), blobs))
                    files.push_back({entry.path().lexically_relative(snapshot).generic_string(), "", 0});
            }
        }
        require(!files.empty(), "Offline cache miss: no downloaded GGUF");
        return snapshot_path(snapshot, select(files, options.file, quant).name).string();
    }
    directories(root);
    const auto lock = root / ".clap-download.lock";
    safe_path(lock);
    require(fs::create_directory(lock), "Another download owns this repository cache (remove stale .clap-download.lock after a crash)");
    struct Unlock { fs::path path; ~Unlock() { std::error_code ec; fs::remove(path, ec); } } unlock{lock};
    std::string endpoint = env("HF_ENDPOINT");
    if (endpoint.empty()) endpoint = "https://huggingface.co";
    while (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
    require(parse_url(endpoint).target == "/", "HF_ENDPOINT must be an HTTP(S) origin");
    const auto token = options.token.empty() ? env("HF_TOKEN") : options.token;
    require(token.find_first_of("\r\n") == std::string::npos, "Invalid HF token");
    Http http(endpoint, token);
    const auto info = json::parse(http.metadata(endpoint + "/api/models/" + repo + "/revision/main")->body);
    const std::string commit = info.at("sha");
    require(hex(commit, 40), "Invalid immutable Hugging Face commit");
    std::vector<File> files;
    std::string url = endpoint + "/api/models/" + repo + "/tree/" + commit + "?recursive=true&limit=1000";
    std::set<std::string> visited;
    while (!url.empty()) {
        require(visited.size() < 1000 && visited.insert(url).second, "Invalid tree pagination");
        require(parse_url(url).origin == parse_url(endpoint).origin, "Cross-origin tree pagination refused");
        auto r = http.metadata(url);
        const auto rows = json::parse(r->body);
        require(rows.is_array(), "Invalid repository tree");
        for (const auto &row : rows) {
            if (row.value("type", "") != "file") continue;
            File f; f.name = row.at("path"); safe_name(f.name);
            if (!gguf(f.name)) continue;
            f.size = row.at("size").get<uint64_t>();
            f.oid = row.contains("lfs") && row["lfs"].is_object() ? row["lfs"].at("oid").get<std::string>() : row.at("oid").get<std::string>();
            require((hex(f.oid, 40) || hex(f.oid, 64)) && f.size > 0, "Invalid model blob metadata");
            files.push_back(f);
        }
        url.clear();
        std::smatch m;
        const auto link = r->get_header_value("Link");
        if (std::regex_search(link, m, std::regex("<([^>]+)>;\\s*rel=\"next\""))) url = next_url(endpoint + "/", m[1]);
    }
    const auto file = select(files, options.file, quant);
    directories(blobs);
    const auto snapshot = snapshot_path(root / "snapshots" / commit, file.name);
    if (!cached(snapshot, blobs, file.size)) {
        std::cerr << "Downloading " << repo << '/' << file.name << " (" << file.size << " bytes)\n";
        download(http, endpoint + '/' + repo + "/resolve/" + commit + '/' + encode(file.name), file, blobs / file.oid);
        directories(snapshot.parent_path());
        require(!fs::exists(snapshot) && !fs::is_symlink(fs::symlink_status(snapshot)), "Invalid existing snapshot file; remove it before retrying");
        const auto tmp = fs::path(snapshot.string() + ".downloadInProgress");
        const auto blob = fs::absolute(blobs / file.oid).lexically_normal();
        // We hold the repository lock. Inspect the link itself, never resolve an
        // arbitrary target; only our selected, validated blob may be published.
        const auto status = fs::symlink_status(tmp);
        if (fs::is_symlink(status)) {
            const auto target = fs::read_symlink(tmp);
            require(target == blob.lexically_relative(snapshot.parent_path()),
                    "Unsafe temporary snapshot symlink: " + tmp.string());
        } else if (fs::exists(status)) {
            safe_path(tmp);
            require(fs::is_regular_file(status) && fs::equivalent(tmp, blob),
                    "Unsafe temporary snapshot file: " + tmp.string());
        } else {
            safe_path(tmp);
            std::error_code link_error;
            fs::create_symlink(blob.lexically_relative(snapshot.parent_path()), tmp, link_error);
            if (link_error) {
                // Native Windows commonly denies symlinks without Developer Mode
                // or elevation. Hard links preserve the same blob/snapshot layout.
                std::error_code hardlink_error;
                fs::create_hard_link(blob, tmp, hardlink_error);
                require(!hardlink_error, "Cannot publish snapshot (symlink: " + link_error.message() +
                        "; hard link: " + hardlink_error.message() + "): " + tmp.string());
            }
        }
        fs::rename(tmp, snapshot);
        std::cerr << "Downloaded " << file.size << " bytes\n";
    }
    publish_text(root / "refs/main", commit);
    return snapshot.string();
}

std::vector<std::string> hf_arguments(int argc, char **argv) {
    std::vector<std::string> args;
    bool literal = false;
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (!literal) for (const auto &alias : {std::pair<const char *, const char *>{"-hf", "--hf-repo"}, {"-hff", "--hf-file"}, {"-hft", "--hf-token"}}) {
            const std::string key = alias.first;
            if (arg == key || arg.rfind(key + "=", 0) == 0) { arg.replace(0, key.size(), alias.second); break; }
        }
        if (arg == "--") literal = true;
        args.push_back(std::move(arg));
    }
    return args;
}
} // namespace clap
