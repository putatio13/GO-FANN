#pragma once

// Host-observed spans. Capture does not add device synchronization.
// Finish after the measured iteration and all its workers have joined.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <nvtx3/nvToolsExt.h>

namespace go_fann::performance_trace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;

inline int64_t ns(Time time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        time.time_since_epoch()).count();
}

struct Event {
    std::string name;
    uintptr_t context;
    size_t thread;
    int64_t start_ns;
    int64_t end_ns;
    double value;
    bool counter;
};

struct Recorder {
    std::atomic<bool> active{false};
    std::mutex mutex;
    std::vector<Event> events;
    std::set<std::pair<std::string, int>> completed;
    std::string path, backend, point_id;
    int output_fd = -1;
    int queue = 0;
    int run = 0;
    uint64_t generation = 0;
    size_t open_scopes = 0;
    std::atomic<bool> failed{false};
    bool poisoned = false;
    ~Recorder() { if (output_fd >= 0) ::close(output_fd); }
};

inline Recorder& recorder() {
    static Recorder instance;
    return instance;
}

inline bool enabled() {
    return recorder().active.load(std::memory_order_relaxed);
}

inline bool point_matches(const std::string& point, const std::string& backend, int queue) {
    const std::string prefix = backend == "pq" ? "b" : "rabitq_b";
    const auto digits = [](const std::string& value) {
        return !value.empty() && value.find_first_not_of("0123456789") == std::string::npos;
    };
    const size_t q = point.find("_q", prefix.size());
    const size_t f = q == std::string::npos ? q : point.find("_f", q + 2);
    return point.compare(0, prefix.size(), prefix) == 0 && q != std::string::npos &&
        f != std::string::npos && [&] {
            const std::string bonus = point.substr(prefix.size(), q - prefix.size());
            const size_t dot = bonus.find('.');
            return dot == std::string::npos ? digits(bonus) :
                digits(bonus.substr(0, dot)) && digits(bonus.substr(dot + 1));
        }() &&
        point.substr(q + 2, f - q - 2) == std::to_string(queue) &&
        digits(point.substr(f + 2)) && point[f + 2] != '0';
}

inline bool begin(const char* backend, const std::string& point_id,
                  const char* phase, int queue, int run) {
    const char* path = std::getenv("GO_FANN_PERFORMANCE_TRACE");
    if (path == nullptr || *path == '\0') return false;
    auto& state = recorder();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.active.load()) throw std::logic_error("nested performance trace iteration");
    if (state.poisoned) throw std::runtime_error("performance trace has an earlier write failure");
    if (backend == nullptr || (std::strcmp(backend, "pq") && std::strcmp(backend, "rabitq")) ||
        phase == nullptr || std::strcmp(phase, "timed") || queue <= 0 || run < 0 ||
        !point_matches(point_id, backend, queue)) {
        throw std::invalid_argument("invalid performance trace point identity");
    }
    if (state.completed.count({point_id, run})) {
        throw std::logic_error("duplicate completed performance trace point/run");
    }
    if (state.output_fd >= 0 && state.path != path) {
        throw std::logic_error("performance trace destination cannot change within a process");
    }
    if (state.output_fd < 0) {
        state.path = path;
        state.output_fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (state.output_fd < 0) {
            throw std::runtime_error("cannot create fresh performance trace: " +
                                     std::string(std::strerror(errno)));
        }
    }
    state.backend = backend;
    state.point_id = point_id;
    state.queue = queue;
    state.run = run;
    state.events.clear();
    state.open_scopes = 0;
    state.failed.store(false);
    ++state.generation;
    state.active.store(true, std::memory_order_relaxed);
    return true;
}

inline void append_locked(Recorder& state, const char* name, const void* context,
                          Time start, Time end, double value, bool counter) {
    const auto first = ns(start), last = ns(end);
    if (name == nullptr || *name == '\0' || first < 0 || last < first ||
        !std::isfinite(value) || value < 0) {
        state.failed.store(true);
        return;
    }
    state.events.push_back({name, reinterpret_cast<uintptr_t>(context),
        std::hash<std::thread::id>{}(std::this_thread::get_id()), first, last, value, counter});
}

inline void span(const char* name, const void* context, Time start, Time end) noexcept {
    if (!enabled()) return;
    auto& state = recorder();
    try {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (state.active.load()) append_locked(state, name, context, start, end, 0.0, false);
    } catch (...) { state.failed.store(true); }
}

inline void value(const char* name, const void* context, double value) noexcept {
    if (!enabled()) return;
    const auto now = Clock::now();
    auto& state = recorder();
    try {
        std::lock_guard<std::mutex> guard(state.mutex);
        if (state.active.load()) append_locked(state, name, context, now, now, value, true);
    } catch (...) { state.failed.store(true); }
}

class Scope {
public:
    Scope(const char* name, const void* context) : name_(name), context_(context) {
        if (!enabled()) return;
        auto& state = recorder();
        {
            std::lock_guard<std::mutex> guard(state.mutex);
            if (!state.active.load()) return;
            generation_ = state.generation;
            ++state.open_scopes;
            active_ = true;
        }
        nvtxRangePushA(name_);
        start_ = Clock::now();
    }
    ~Scope() noexcept {
        if (!active_) return;
        const auto end = Clock::now();
        nvtxRangePop();
        auto& state = recorder();
        try {
            std::lock_guard<std::mutex> guard(state.mutex);
            if (generation_ != state.generation || !state.active.load()) return;
            --state.open_scopes;
            append_locked(state, name_, context_, start_, end, 0.0, false);
        } catch (...) { state.failed.store(true); }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    const char* name_;
    const void* context_;
    bool active_ = false;
    uint64_t generation_ = 0;
    Time start_;
};

inline void quoted(std::ostream& output, const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') output << '\\' << static_cast<char>(c);
        else if (c < 0x20) output << "\\u00" << hex[c >> 4] << hex[c & 15];
        else output << static_cast<char>(c);
    }
    output << '"';
}

inline void identity(std::ostream& output, const Recorder& state) {
    output << "{\"schema_version\":1,\"backend\":";
    quoted(output, state.backend);
    output << ",\"point_id\":";
    quoted(output, state.point_id);
    output << ",\"phase\":\"timed\",\"queue\":" << state.queue << ",\"run\":" << state.run;
}

inline void write_all(Recorder& state, const std::string& data) {
    size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t written = ::write(state.output_fd, data.data() + offset, data.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            state.poisoned = true;
            throw std::runtime_error("cannot write performance trace");
        }
        offset += static_cast<size_t>(written);
    }
}

inline void cancel() noexcept {
    auto& state = recorder();
    try {
        std::lock_guard<std::mutex> guard(state.mutex);
        state.active.store(false, std::memory_order_relaxed);
        state.events.clear();
        state.open_scopes = 0;
    } catch (...) { state.active.store(false, std::memory_order_relaxed); }
}

inline void finish() {
    auto& state = recorder();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!state.active.exchange(false, std::memory_order_relaxed)) return;
    if (state.failed.load() || state.open_scopes || state.events.empty()) {
        state.events.clear();
        throw std::runtime_error("cannot complete performance trace: invalid, unfinished or empty events");
    }
    std::ostringstream output;
    output << std::setprecision(17);
    for (const auto& event : state.events) {
        identity(output, state);
        output << ",\"kind\":\"" << (event.counter ? "counter" : "span") << "\",\"name\":";
        quoted(output, event.name);
        output << ",\"context\":" << event.context << ",\"thread\":" << event.thread
               << ",\"start_ns\":" << event.start_ns << ",\"end_ns\":" << event.end_ns
               << ",\"value\":" << event.value << "}\n";
    }
    std::ostringstream completion;
    identity(completion, state);
    completion << ",\"kind\":\"complete\",\"event_count\":" << state.events.size() << "}\n";
    const auto event_data = output.str(), completion_data = completion.str();
    state.completed.insert({state.point_id, state.run});
    write_all(state, event_data);
    write_all(state, completion_data);
    state.events.clear();
}

class Capture {
public:
    Capture(const char* backend, const std::string& point_id, const char* phase, int queue, int run)
        : active_(begin(backend, point_id, phase, queue, run)) {}
    ~Capture() noexcept { if (active_) cancel(); }
    void finish() {
        if (active_) { performance_trace::finish(); active_ = false; }
    }
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
private:
    bool active_;
};
}  // namespace go_fann::performance_trace
