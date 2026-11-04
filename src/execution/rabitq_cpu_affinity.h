#pragma once

#include <pthread.h>
#include <sched.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

namespace go_fann::backend_rabitq {

// Explicit pool-worker placement. Coordinators and mailbox responders retain
// their own affinity. An empty setting keeps inherited scheduler behavior.
class RabitqCpuAffinity {
public:
    explicit RabitqCpuAffinity(const char* cpus) : text_(cpus == nullptr ? "" : cpus)
    {
        CPU_ZERO(&mask_);
        size_t pos = 0;
        while (pos < text_.size()) {
            const int first = parse_cpu(pos);
            int last = first;
            if (pos < text_.size() && text_[pos] == '-') {
                ++pos;
                last = parse_cpu(pos);
            }
            if (last < first) throw std::invalid_argument("rerank CPU range is reversed");
            for (int cpu = first; cpu <= last; ++cpu) CPU_SET(cpu, &mask_);
            if (pos == text_.size()) break;
            if (text_[pos++] != ',' || pos == text_.size()) {
                throw std::invalid_argument("invalid rerank CPU list: " + text_);
            }
        }
    }

    bool enabled() const { return !text_.empty(); }
    const std::string& text() const { return text_; }

    void bind_current_thread() const
    {
        if (!enabled()) return;
        int error = pthread_setaffinity_np(pthread_self(), sizeof(mask_), &mask_);
        if (error != 0) throw std::system_error(error, std::generic_category(), "rerank CPU affinity");
        cpu_set_t actual;
        error = pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual);
        if (error != 0) throw std::system_error(error, std::generic_category(), "read rerank CPU affinity");
        // Linux may silently intersect the requested mask with a cpuset cgroup.
        if (!CPU_EQUAL(&mask_, &actual)) {
            throw std::runtime_error("rerank CPU affinity was restricted: " + text_);
        }
        error = pthread_setname_np(pthread_self(), "fa-rerank");
        if (error != 0) throw std::system_error(error, std::generic_category(), "name rerank worker");
    }

private:
    int parse_cpu(size_t& pos) const
    {
        if (pos == text_.size() || text_[pos] < '0' || text_[pos] > '9') {
            throw std::invalid_argument("invalid rerank CPU list: " + text_);
        }
        int cpu = 0;
        do {
            cpu = cpu * 10 + (text_[pos++] - '0');
            if (cpu >= CPU_SETSIZE) {
                throw std::invalid_argument("rerank CPU exceeds CPU_SETSIZE");
            }
        } while (pos < text_.size() && text_[pos] >= '0' && text_[pos] <= '9');
        return cpu;
    }

    std::string text_;
    cpu_set_t mask_;
};

}  // namespace go_fann::backend_rabitq
