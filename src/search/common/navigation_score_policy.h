#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace go_fann::navigation_score_policy {

// Only the maintained native memory search changes score semantics. Other
// workers/transports retain their legacy score and bonus configuration.
inline bool uses_raw_scores(std::string_view backend, std::string_view operation,
                            std::string_view worker, std::string_view transport,
                            std::string_view graph_storage)
{
    return (backend == "pq" || backend == "rabitq") &&
        (operation == "search" || operation == "benchmark") &&
        worker == "coupled" && transport == "mailbox" && graph_storage == "memory";
}

inline void validate_bonus_axis(bool raw_scores, std::size_t count)
{
    if (raw_scores && count > 1) {
        throw std::runtime_error(
            "benchmark-quick-filter-bonuses is obsolete on native coupled memory search; "
            "multiple values would repeat identical scoring. Keep at most one legacy "
            "value for artifact naming, or omit the axis");
    }
}

}  // namespace go_fann::navigation_score_policy
