// Generate short-lived TLS fixtures with the same OpenSSL used by the downloader.
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char *operation) {
    if (!ok) throw std::runtime_error(std::string("HTTPS certificate fixture failed: ") + operation);
}
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using Cert = std::unique_ptr<X509, decltype(&X509_free)>;
Key key() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
        EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr), EVP_PKEY_CTX_free);
    require(bool(ctx), "EVP_PKEY_CTX_new_id");
    require(EVP_PKEY_keygen_init(ctx.get()) == 1, "EVP_PKEY_keygen_init");
    require(EVP_PKEY_CTX_set_rsa_keygen_bits(ctx.get(), 2048) == 1, "EVP_PKEY_CTX_set_rsa_keygen_bits");
    EVP_PKEY *result = nullptr;
    require(EVP_PKEY_keygen(ctx.get(), &result) == 1, "EVP_PKEY_keygen");
    return Key(result, EVP_PKEY_free);
}
void extension(X509 *cert, X509 *issuer, int nid, const char *value) {
    std::cerr << "Adding extension " << OBJ_nid2sn(nid) << ": " << value << '\n';
    X509V3_CTX ctx;
    X509V3_set_ctx(&ctx, issuer, cert, nullptr, nullptr, 0);
    std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> ext(
        X509V3_EXT_conf_nid(nullptr, &ctx, nid, value), X509_EXTENSION_free);
    require(bool(ext), "X509V3_EXT_conf_nid");
    require(X509_add_ext(cert, ext.get(), -1) == 1, "X509_add_ext");
}
Cert certificate(EVP_PKEY *key, X509 *issuer, EVP_PKEY *issuer_key) {
    Cert cert(X509_new(), X509_free);
    require(bool(cert), "X509_new");
    require(X509_set_version(cert.get(), 2) == 1, "X509_set_version");
    require(ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), issuer ? 2 : 1) == 1, "ASN1_INTEGER_set");
    require(X509_gmtime_adj(X509_getm_notBefore(cert.get()), -3600) != nullptr, "X509_gmtime_adj(notBefore)");
    require(X509_gmtime_adj(X509_getm_notAfter(cert.get()), 86400) != nullptr, "X509_gmtime_adj(notAfter)");
    require(X509_set_pubkey(cert.get(), key) == 1, "X509_set_pubkey");
    auto *name = X509_get_subject_name(cert.get());
    require(X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
        reinterpret_cast<const unsigned char *>(issuer ? "HF test server" : "HF test CA"),
        -1, -1, 0) == 1, "X509_NAME_add_entry_by_txt");
    require(X509_set_issuer_name(cert.get(), issuer ? X509_get_subject_name(issuer) : name) == 1, "X509_set_issuer_name");
    extension(cert.get(), issuer ? issuer : cert.get(), NID_basic_constraints,
              issuer ? "critical,CA:FALSE" : "critical,CA:TRUE");
    extension(cert.get(), issuer ? issuer : cert.get(), NID_key_usage,
              issuer ? "critical,digitalSignature,keyEncipherment" : "critical,keyCertSign,cRLSign");
    if (issuer) {
        extension(cert.get(), issuer, NID_ext_key_usage, "serverAuth");
        extension(cert.get(), issuer, NID_subject_alt_name, "IP:127.0.0.1");
    }
    require(X509_sign(cert.get(), issuer_key ? issuer_key : key, EVP_sha256()) > 0, "X509_sign");
    return cert;
}
void write(const std::filesystem::path &path, X509 *cert, EVP_PKEY *key = nullptr) {
    std::cerr << "Writing " << path.string() << '\n';
    const auto close = [](FILE *file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(close)> file(std::fopen(path.string().c_str(), "wb"), close);
    if (!file) throw std::runtime_error("fopen " + path.string() + ": " + std::strerror(errno));
    require((key ? PEM_write_PrivateKey(file.get(), key, nullptr, nullptr, 0, nullptr, nullptr)
                 : PEM_write_X509(file.get(), cert)) == 1, key ? "PEM_write_PrivateKey" : "PEM_write_X509");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected fixture directory");
        const std::filesystem::path directory(argv[1]);
        std::cerr << "TLS fixture: " << OpenSSL_version(OPENSSL_VERSION) << '\n';
        std::cerr << "Generating CA key\n";
        auto ca_key = key();
        std::cerr << "Generating CA certificate\n";
        auto ca = certificate(ca_key.get(), nullptr, nullptr);
        std::cerr << "Generating server key\n";
        auto server_key = key();
        std::cerr << "Generating server certificate\n";
        auto server = certificate(server_key.get(), ca.get(), ca_key.get());
        write(directory / "ca.pem", ca.get());
        write(directory / "server.pem", server.get());
        write(directory / "server-key.pem", nullptr, server_key.get());
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        ERR_print_errors_fp(stderr);
        return 1;
    }
}
