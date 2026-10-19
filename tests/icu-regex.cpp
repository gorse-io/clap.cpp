#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/uclean.h>
#include <unicode/unistr.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        UErrorCode status = U_ZERO_ERROR;
        u_init(&status);
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
        if (!u_isUWhiteSpace(0x3000) || u_charType(0x4f60) != U_OTHER_LETTER ||
            u_charType(0x0661) != U_DECIMAL_DIGIT_NUMBER)
            throw std::runtime_error("ICU Unicode property data unavailable");
        const auto expression = icu::UnicodeString::fromUTF8(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
        std::unique_ptr<icu::RegexPattern> pattern(icu::RegexPattern::compile(expression, 0, status));
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
        // Use UTF-8 escapes so this source also compiles under MSVC's default code page.
        const std::string chinese = "\xe4\xbd\xa0\xe5\xa5\xbd";
        const std::string cafe = "caf\xc3\xa9";
        const std::string music = "\xf0\x9f\x8e\xb5";
        const auto input = icu::UnicodeString::fromUTF8("I'm " + chinese + " " + cafe + "\n" + music);
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
        std::vector<std::string> actual;
        while (matcher->find(status)) {
            std::string token; matcher->group(status).toUTF8String(token); actual.push_back(token);
        }
        const std::vector<std::string> expected{"I", "'m", " " + chinese, " " + cafe, "\n", music};
        if (U_FAILURE(status) || actual != expected) throw std::runtime_error("Unicode regex pre-tokenization mismatch");
        std::cout << "ICU initialization, Unicode properties and CLAP regex passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
