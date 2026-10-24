// Generate short-lived TLS fixtures with the same OpenSSL used by the downloader.
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool ok) {
    if (!ok) throw std::runtime_error("Cannot generate HTTPS test certificates");
}
using Key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using Cert = std::unique_ptr<X509, decltype(&X509_free)>;
Key key() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
        EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr), EVP_PKEY_CTX_free);
    require(bool(ctx));
    require(EVP_PKEY_keygen_init(ctx.get()) == 1);
    require(EVP_PKEY_CTX_set_rsa_keygen_bits(ctx.get(), 2048) == 1);
    EVP_PKEY *result = nullptr;
    require(EVP_PKEY_keygen(ctx.get(), &result) == 1);
    return Key(result, EVP_PKEY_free);
}
void extension(X509 *cert, X509 *issuer, int nid, const char *value) {
    X509V3_CTX ctx;
    X509V3_set_ctx(&ctx, issuer, cert, nullptr, nullptr, 0);
    std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> ext(
        X509V3_EXT_conf_nid(nullptr, &ctx, nid, value), X509_EXTENSION_free);
    require(bool(ext));
    require(X509_add_ext(cert, ext.get(), -1) == 1);
}
Cert certificate(EVP_PKEY *key, X509 *issuer, EVP_PKEY *issuer_key) {
    Cert cert(X509_new(), X509_free);
    require(bool(cert));
    require(X509_set_version(cert.get(), 2) == 1);
    require(ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), issuer ? 2 : 1) == 1);
    require(X509_gmtime_adj(X509_getm_notBefore(cert.get()), -3600) != nullptr);
    require(X509_gmtime_adj(X509_getm_notAfter(cert.get()), 86400) != nullptr);
    require(X509_set_pubkey(cert.get(), key) == 1);
    auto *name = X509_get_subject_name(cert.get());
    require(X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
        reinterpret_cast<const unsigned char *>(issuer ? "HF test server" : "HF test CA"),
        -1, -1, 0) == 1);
    require(X509_set_issuer_name(cert.get(), issuer ? X509_get_subject_name(issuer) : name) == 1);
    extension(cert.get(), issuer ? issuer : cert.get(), NID_basic_constraints,
              issuer ? "critical,CA:FALSE" : "critical,CA:TRUE");
    extension(cert.get(), issuer ? issuer : cert.get(), NID_key_usage,
              issuer ? "critical,digitalSignature,keyEncipherment" : "critical,keyCertSign,cRLSign");
    if (issuer) {
        extension(cert.get(), issuer, NID_ext_key_usage, "serverAuth");
        extension(cert.get(), issuer, NID_subject_alt_name, "IP:127.0.0.1");
    }
    require(X509_sign(cert.get(), issuer_key ? issuer_key : key, EVP_sha256()) > 0);
    return cert;
}
void write(const std::filesystem::path &path, X509 *cert, EVP_PKEY *key = nullptr) {
    const auto close = [](FILE *file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(close)> file(std::fopen(path.string().c_str(), "wb"), close);
    require(bool(file));
    require((key ? PEM_write_PrivateKey(file.get(), key, nullptr, nullptr, 0, nullptr, nullptr)
                 : PEM_write_X509(file.get(), cert)) == 1);
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected fixture directory");
        const std::filesystem::path directory(argv[1]);
        auto ca_key = key();
        auto ca = certificate(ca_key.get(), nullptr, nullptr);
        auto server_key = key();
        auto server = certificate(server_key.get(), ca.get(), ca_key.get());
        write(directory / "ca.pem", ca.get());
        write(directory / "server.pem", server.get());
        write(directory / "server-key.pem", nullptr, server_key.get());
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
