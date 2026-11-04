#pragma once
#include "predicates/gpu_predicate_index.h"

namespace go_fann::gpu_predicate {
// Identity groups preserve exact original-label cardinalities, including labels
// whose postings live in the low-frequency index. label_size alone does not:
// it is zero for those low-frequency groups. The optional original cardinality
// vector is populated lazily by some planners, and is not required here.
template<class Metadata, class Layout>
void validate_identity_cardinalities(const HostIndex& index, const Metadata& metadata,
                                     const Layout& layout, uint32_t original_labels)
{
    require(metadata.original_label_count == original_labels &&
            metadata.label_count == original_labels &&
            layout.label_to_group.size() == original_labels &&
            layout.groups.size() == metadata.label_count &&
            metadata.group_posting_size.size() == metadata.label_count &&
            metadata.cat_freq.size() == metadata.label_count &&
            metadata.label_size.size() == metadata.label_count,
            "identity metadata dimensions mismatch");
    require(metadata.original_label_cardinality.empty() ||
            metadata.original_label_cardinality.size() == original_labels,
            "optional original cardinalities have wrong dimensions");
    size_t handle = 0;
    for (uint32_t label = 0; label < original_labels; ++label) {
        const auto group = layout.label_to_group[label];
        require(group < layout.groups.size() && layout.groups[group].size() == 1 &&
                layout.groups[group][0] == label, "index requires singleton identity groups");
        const auto cardinality = metadata.group_posting_size[group];
        require(cardinality == metadata.cat_freq[group] && cardinality <= index.rows,
                "native group cardinality mismatch");
        require(metadata.original_label_cardinality.empty() ||
                metadata.original_label_cardinality[label] == cardinality,
                "native original and singleton group cardinalities disagree");
        const bool high = cardinality > index.threshold;
        require(metadata.label_size[group] == (high ? cardinality : 0),
                "native graph eligibility/cardinality mismatch");
        if (!high) continue;
        require(handle < index.labels.size() && index.labels[handle] == label &&
                index.cardinalities[handle] == cardinality,
                "high-frequency label set/cardinality mismatch");
        ++handle;
    }
    require(handle == index.labels.size(), "index contains extra high-frequency labels");
}
}
