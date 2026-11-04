#include "data/dataloader.h"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    HybridQueryClauseTable table;
    table.virtual_descriptors = true;
    table.query_indptr = {0}; table.descriptor_indptr = {0}; table.label_indptr = {0};
    HybridClauseExpression unary;
    unary.active = unary.unary = true;
    unary.atoms[0] = {0, 3, false, true};
    table.atom_labels = {2, 4, 6}; table.atom_source_tiers = {1, 1, 1}; table.atom_relevance_ranks = {3, 1, 2};
    table.expressions.push_back(unary);
    table.finish_clause_row(); ++table.rows;
    // Unsupported/eager rows can share the same global logical clause domain.
    table.labels = {8, 9}; table.source_tiers = {0, 0}; table.relevance_ranks = {0, 0};
    table.label_indptr = {0, 1, 2}; table.top_branch_ids = {7, 8}; table.uses_original_range = {0, 1};
    table.finish_clause_row(); ++table.rows;
    HybridClauseExpression binary;
    binary.active = true;
    binary.atoms[0] = {3, 0, true, false};
    binary.atoms[1] = {3, 2, false, true};
    binary.branch_ids = {11, 12};
    table.atom_labels.insert(table.atom_labels.end(), {4, 6});
    table.atom_source_tiers.insert(table.atom_source_tiers.end(), {1, 1});
    table.atom_relevance_ranks.insert(table.atom_relevance_ranks.end(), {0, 2});
    table.expressions.push_back(binary);
    table.finish_clause_row(); ++table.rows;
    table.finish_clause_row(); ++table.rows; // Empty row must not claim a clause.
    require(table.descriptor_shape_valid() && table.clause_count() == 8, "mixed clause shape");
    require(table.descriptor_indptr == std::vector<uint64_t>({0, 0, 2, 2, 2}), "mixed eager offsets");
    require(table.query_indptr == std::vector<uint64_t>({0, 3, 5, 8, 8}), "logical offsets");
    for (uint32_t id = 0; id < 3; ++id) {
        const auto labels = table.clause_labels(id);
        require(labels.valid && labels.count == 1 && labels.begin()[0] == 2 + id * 2, "unary labels");
        require(labels.tier(0) == 1 && labels.rank(0) == std::array<uint32_t, 3>{3, 1, 2}[id], "unary occurrences");
        require(table.clause_branch(id) == id && !table.clause_uses_range(id), "unary branch");
    }
    require(table.clause_labels(3).begin()[0] == 8 && table.clause_branch(3) == 7, "eager row in virtual table");
    require(table.clause_labels(4).begin()[0] == 9 && table.clause_uses_range(4), "eager range");
    require(table.clause_labels(5).valid && table.clause_labels(5).count == 0 && table.clause_uses_range(5), "range-only expression");
    require(table.clause_branch(5) == 11 && table.clause_branch(6) == 12, "OR branches");
    require(!table.clause_labels(3, 0).valid && !table.clause_labels(8).valid, "invalid owner/id");
    require(!table.has_clauses(3), "empty row");
    table.descriptor_indptr.pop_back();
    require(!table.descriptor_shape_valid() && !table.clause_labels(0).valid, "invalid storage offsets");

    // Large logical expansions keep one expression and no branch descriptor arrays.
    HybridQueryClauseTable large;
    large.virtual_descriptors = true;
    large.query_indptr = {0}; large.descriptor_indptr = {0}; large.label_indptr = {0};
    HybridClauseExpression conjunction;
    conjunction.active = conjunction.conjunction = true;
    conjunction.atoms[0] = {0, 0, true, false};
    conjunction.atoms[1] = {0, 100000, false, true};
    for (uint32_t i = 0; i < 100000; ++i) {
        large.atom_labels.push_back(i); large.atom_source_tiers.push_back(1); large.atom_relevance_ranks.push_back(i);
    }
    large.expressions.push_back(conjunction); large.finish_clause_row(); ++large.rows;
    require(large.descriptor_shape_valid() && large.clause_count() == 100000 && large.top_branch_ids.empty() &&
        large.uses_original_range.empty() && large.label_indptr.size() == 1, "large expansion stored descriptors");
    for (uint32_t id : {0U, 1U, 50000U, 99999U}) {
        const auto labels = large.clause_labels(id);
        require(labels.valid && labels.count == 1 && labels.begin()[0] == id && labels.rank(0) == id &&
            large.clause_uses_range(id) && large.clause_branch(id) == 0, "large expression semantics");
    }
    std::cout << "Virtual/eager mixed clause access and large expansion passed\n";
}
