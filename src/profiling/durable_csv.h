#pragma once

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

namespace go_fann {

class DurableCsvAppender {
public:
    DurableCsvAppender(const std::string& path, std::string_view header)
        : path_(path)
    {
        fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("unable to open durable CSV " + path + ": " +
                                     std::strerror(errno));
        }
        try {
            append_bytes(header);
            if (header.empty() || header.back() != '\n') {
                append_bytes("\n");
            }
            sync();
        } catch (...) {
            close(fd_);
            fd_ = -1;
            throw;
        }
    }

    ~DurableCsvAppender()
    {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    DurableCsvAppender(const DurableCsvAppender&) = delete;
    DurableCsvAppender& operator=(const DurableCsvAppender&) = delete;

    void append_row(std::string_view row)
    {
        append_bytes(row);
        if (row.empty() || row.back() != '\n') {
            append_bytes("\n");
        }
        sync();
    }

private:
    void append_bytes(std::string_view bytes)
    {
        size_t written = 0;
        while (written < bytes.size()) {
            const ssize_t result = write(fd_, bytes.data() + written, bytes.size() - written);
            if (result < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("unable to write durable CSV " + path_ + ": " +
                                         std::strerror(errno));
            }
            if (result == 0) {
                throw std::runtime_error("short write to durable CSV " + path_);
            }
            written += static_cast<size_t>(result);
        }
    }

    void sync()
    {
        if (fdatasync(fd_) != 0) {
            throw std::runtime_error("unable to sync durable CSV " + path_ + ": " +
                                     std::strerror(errno));
        }
    }

    std::string path_;
    int fd_ = -1;
};

} // namespace go_fann
