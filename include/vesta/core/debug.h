#pragma once

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace vesta::core::debug {
inline void LogMessage(std::string_view message)
{
    const std::string line(message);
    std::fputs(line.c_str(), stderr);
    std::fputc('\n', stderr);
#if defined(_WIN32)
    OutputDebugStringA((line + "\n").c_str());
#endif
}

inline std::string BuildAssertMessage(
    std::string_view expression, std::string_view message, const char* file, int line, const char* function)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << expression;
    if (!message.empty()) {
        stream << " | " << message;
    }
    stream << " | " << file << ':' << line << " | " << function;
    return stream.str();
}

[[noreturn]] inline void Panic(
    std::string_view expression, std::string_view message, const char* file, int line, const char* function)
{
    LogMessage(BuildAssertMessage(expression, message, file, line, function));
#if defined(_MSC_VER)
    __debugbreak();
#endif
    std::abort();
}

inline void ReportAssertionFailure(
    std::string_view expression, std::string_view message, const char* file, int line, const char* function)
{
    LogMessage(BuildAssertMessage(expression, message, file, line, function));
}
} // namespace vesta::core::debug

#if defined(NDEBUG)
#define VESTA_ASSERT(expr, message)                                                                                    \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            ::vesta::core::debug::ReportAssertionFailure(#expr, (message), __FILE__, __LINE__, __func__);            \
        }                                                                                                              \
    } while (false)
#else
#define VESTA_ASSERT(expr, message)                                                                                    \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            ::vesta::core::debug::Panic(#expr, (message), __FILE__, __LINE__, __func__);                             \
        }                                                                                                              \
    } while (false)
#endif

#define VESTA_ASSERT_STATE(expr, message) VESTA_ASSERT((expr), (message))
#define VESTA_UNREACHABLE(message) VESTA_ASSERT(false, (message))
