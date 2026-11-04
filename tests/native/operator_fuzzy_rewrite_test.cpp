#include "execution/async_io.h"
#include "app/config_loader.h"
#include "data/dataloader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

AsyncIoConfig::AsyncIoConfig(uint32_t dim_value,
                             uint32_t degree_value,
                             uint32_t prefetch_hops_value,
                             size_t block_size_value,
                             size_t header_size_value)
    : dim(dim_value),
      degree(degree_value),
      prefetch_hops(prefetch_hops_value),
      block_size(block_size_value),
      header_size(header_size_value)
{
}

void AsyncIoConfig::set_reserve_vector_payload(bool reserve)
{
    reserve_vector_payload = reserve;
}

void build_prefetch_slices(GraphRuntime&, const AsyncIoConfig&)
{
}

namespace {

constexpr std::array<char, 16> kNgramMagic{
    'F', 'A', 'N', 'N', 'S', 'N', 'G', 'R', 'A', 'M', 'I', 'D', 'X', '\0', '\0', '\0'};
constexpr std::array<char, 16> kOperatorMagic{
    'F', 'A', 'N', 'N', 'S', 'O', 'P', 'L', 'E', 'X', 'I', 'D', 'X', '\0', '\0', '\0'};

struct NgramHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t endian;
    uint32_t flags;
    uint64_t label_count;
    uint64_t token_count;
    uint64_t ngram_size;
    uint64_t normalizer_version;
    uint64_t token_text_bytes;
    uint64_t ngram_count;
    uint64_t ngram_text_bytes;
    uint64_t posting_count;
    uint64_t checksum_kind;
    uint64_t reserved;
    uint8_t checksum[32];
};

struct OperatorHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t endian;
    uint32_t flags;
    uint64_t token_count;
    uint64_t prefix_count;
    uint64_t suffix_count;
    uint64_t cost_count;
    uint8_t checksum[32];
};

static_assert(sizeof(NgramHeader) == 144, "ngram fixture header layout mismatch");
static_assert(sizeof(OperatorHeader) == 96, "operator fixture header layout mismatch");

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws_with_message(Function&& function,
                                 const std::string& expected,
                                 const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message);
}

std::string temporary_path(const char* suffix)
{
    char path[] = "/tmp/flashanns_operator_rewrite_test_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    std::remove(path);
    return std::string(path) + suffix;
}

std::vector<std::string> token_ngrams(const std::string& text, uint32_t ngram_size)
{
    if (text.size() <= ngram_size) {
        return {text};
    }
    std::vector<std::string> grams;
    for (size_t offset = 0; offset + ngram_size <= text.size(); ++offset) {
        grams.push_back(text.substr(offset, ngram_size));
    }
    std::sort(grams.begin(), grams.end());
    grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
    return grams;
}

template <typename Value>
void write_values(std::ofstream& output, const std::vector<Value>& values)
{
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(Value)));
    }
}

void write_ngram_fixture(const std::string& path,
                         const std::vector<std::string>& texts,
                         const std::array<uint8_t, 32>& checksum,
                         bool invalid_token_text_offsets = false)
{
    std::vector<uint32_t> token_ids(texts.size());
    std::vector<uint32_t> token_ngram_counts(texts.size());
    std::vector<uint64_t> token_text_offsets{0};
    std::string token_text_payload;
    std::map<std::string, std::vector<uint32_t>> postings;
    for (size_t token_id = 0; token_id < texts.size(); ++token_id) {
        token_ids[token_id] = static_cast<uint32_t>(token_id);
        token_text_payload += texts[token_id];
        token_text_offsets.push_back(token_text_payload.size());
        const std::vector<std::string> grams = token_ngrams(texts[token_id], 3);
        token_ngram_counts[token_id] = static_cast<uint32_t>(grams.size());
        for (const std::string& gram : grams) {
            postings[gram].push_back(static_cast<uint32_t>(token_id));
        }
    }
    if (invalid_token_text_offsets && !token_text_offsets.empty()) {
        ++token_text_offsets.back();
    }

    std::vector<uint64_t> ngram_text_offsets{0};
    std::vector<uint64_t> ngram_offsets{0};
    std::string ngram_text_payload;
    std::vector<uint32_t> posting_ids;
    for (const auto& entry : postings) {
        ngram_text_payload += entry.first;
        ngram_text_offsets.push_back(ngram_text_payload.size());
        posting_ids.insert(posting_ids.end(), entry.second.begin(), entry.second.end());
        ngram_offsets.push_back(posting_ids.size());
    }

    NgramHeader header{};
    std::memcpy(header.magic, kNgramMagic.data(), kNgramMagic.size());
    header.version = 1;
    header.header_bytes = sizeof(header);
    header.endian = 0x01020304u;
    header.label_count = texts.size();
    header.token_count = texts.size();
    header.ngram_size = 3;
    header.normalizer_version = 1;
    header.token_text_bytes = token_text_payload.size();
    header.ngram_count = postings.size();
    header.ngram_text_bytes = ngram_text_payload.size();
    header.posting_count = posting_ids.size();
    header.checksum_kind = 1;
    std::copy(checksum.begin(), checksum.end(), header.checksum);

    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "failed to create ngram fixture");
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_values(output, token_ids);
    write_values(output, token_ngram_counts);
    write_values(output, token_text_offsets);
    output.write(token_text_payload.data(), static_cast<std::streamsize>(token_text_payload.size()));
    write_values(output, ngram_text_offsets);
    write_values(output, ngram_offsets);
    output.write(ngram_text_payload.data(), static_cast<std::streamsize>(ngram_text_payload.size()));
    write_values(output, posting_ids);
    require(output.good(), "failed to write ngram fixture");
}

std::array<uint8_t, 32> checksum_from_hex(const char* hex)
{
    std::array<uint8_t, 32> checksum{};
    for (size_t idx = 0; idx < checksum.size(); ++idx) {
        const std::string byte(hex + idx * 2, 2);
        checksum[idx] = static_cast<uint8_t>(std::stoul(byte, nullptr, 16));
    }
    return checksum;
}

void write_operator_fixture(const std::string& path,
                            const std::vector<uint32_t>& prefix,
                            const std::vector<uint32_t>& suffix,
                            const std::vector<uint32_t>& costs,
                            const std::array<uint8_t, 32>& checksum,
                            bool valid_magic = true)
{
    OperatorHeader header{};
    if (valid_magic) {
        std::memcpy(header.magic, kOperatorMagic.data(), kOperatorMagic.size());
    } else {
        std::memcpy(header.magic, "BADOPERATOR", 11);
    }
    header.version = 1;
    header.header_bytes = sizeof(header);
    header.endian = 0x01020304u;
    header.flags = 0x7u;
    header.token_count = costs.size();
    header.prefix_count = prefix.size();
    header.suffix_count = suffix.size();
    header.cost_count = costs.size();
    std::copy(checksum.begin(), checksum.end(), header.checksum);

    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "failed to create operator fixture");
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_values(output, prefix);
    write_values(output, suffix);
    write_values(output, costs);
    require(output.good(), "failed to write operator fixture");
}

void require_labels(const FuzzyOperatorRewriteTestResult& result,
                    std::vector<uint32_t> expected,
                    bool expected_fallback,
                    const std::string& context)
{
    require(result.operator_labels == expected, context + " operator labels mismatch");
    require(result.baseline_labels == expected, context + " baseline labels mismatch");
    require(result.operator_labels == result.baseline_labels, context + " parity mismatch");
    require(result.used_fallback == expected_fallback, context + " fallback state mismatch");
}

void require_occurrence_metadata_aligned(const FuzzyOperatorRewriteTestResult& result,
                                         const std::string& context)
{
    require(result.operator_source_tiers.size() == result.operator_labels.size(),
            context + " source tiers are not label-aligned");
    require(result.operator_relevance_ranks.size() == result.operator_labels.size(),
            context + " relevance ranks are not label-aligned");
}

}  // namespace

int main()
{
    const std::vector<std::string> texts{
        "alpha", "alphabet", "alpine", "beta", "theta", "meta", "betamax",
        "xalpha", "al", "abcd", "xabcd", "abcbcd", "zzabcd"};
    const std::vector<uint32_t> prefix{11, 9, 8, 0, 1, 2, 3, 6, 5, 4, 10, 7, 12};
    const std::vector<uint32_t> suffix{0, 7, 3, 4, 5, 9, 10, 12, 11, 2, 8, 1, 6};
    const std::vector<uint32_t> costs{10, 40, 30, 97, 96, 95, 94, 93, 20, 91, 90, 89, 88};
    const auto valid_checksum = checksum_from_hex(
        "fb2604d579202cf6bee2f2efff9e27091f6714db6763676e11437915ff835b92");
    const auto invalid_permutation_checksum = checksum_from_hex(
        "30242d5867d7be2ad4b41da3153eaea820a23011990d530b167df88ef81a7d0b");
    const auto invalid_order_checksum = checksum_from_hex(
        "6be4f5f6960c2f7e77b31767f11dab5105335332e22a147ac4d0c8533b6ecfe7");
    const auto ngram_checksum = checksum_from_hex(
        "3c44e7e261ee9e2f03b31ec613eddde0e77e0d7ca81b277bf6dd16ab70fc056b");
    const auto short_ngram_checksum = checksum_from_hex(
        "4d2a276cbe8941c80ebc3b90fc2375c6a7fa97b3cac7fd333d7f7e01722527a3");
    const auto invalid_offsets_checksum = checksum_from_hex(
        "e5419e619c3e68d6c82a361f36bfc7bb4e31c3a9c1cf2f5740da90752fd35d85");
    const auto nested_ngram_checksum = checksum_from_hex(
        "0f816955801e329ca97671c93cd2b46bd3a25721c1d36bd52d6ff09c5abb1ab0");
    const auto nested_operator_checksum = checksum_from_hex(
        "6360bb7e16975ea8622515555f39950bf40f99316ce639eeafea91f4c984ba46");

    const std::string ngram_path = temporary_path(".ngram.bin");
    const std::string short_ngram_path = temporary_path(".short.ngram.bin");
    const std::string bad_ngram_checksum_path = temporary_path(".bad-ngram-checksum.bin");
    const std::string bad_ngram_size_path = temporary_path(".bad-ngram-size.bin");
    const std::string bad_ngram_offsets_path = temporary_path(".bad-ngram-offsets.bin");
    const std::string operator_path = temporary_path(".operator.bin");
    const std::string bad_magic_path = temporary_path(".bad-magic.bin");
    const std::string bad_checksum_path = temporary_path(".bad-checksum.bin");
    const std::string bad_permutation_path = temporary_path(".bad-permutation.bin");
    const std::string bad_order_path = temporary_path(".bad-order.bin");
    const std::string bad_size_path = temporary_path(".bad-size.bin");
    const std::string nested_ngram_path = temporary_path(".nested.ngram.bin");
    const std::string nested_operator_path = temporary_path(".nested.operator.bin");
    const std::string nested_config_path = temporary_path(".nested.config.json");
    const std::string literal_config_path = temporary_path(".literal.config.json");

    write_ngram_fixture(ngram_path, texts, ngram_checksum);
    write_ngram_fixture(short_ngram_path,
                        std::vector<std::string>(texts.begin(), texts.end() - 1),
                        short_ngram_checksum);
    write_ngram_fixture(bad_ngram_checksum_path, texts, {});
    write_ngram_fixture(bad_ngram_size_path, texts, ngram_checksum);
    {
        std::ofstream output(bad_ngram_size_path, std::ios::binary | std::ios::app);
        output.put('\0');
    }
    write_ngram_fixture(
        bad_ngram_offsets_path, texts, invalid_offsets_checksum, true);
    write_operator_fixture(operator_path, prefix, suffix, costs, valid_checksum);
    write_operator_fixture(bad_magic_path, prefix, suffix, costs, valid_checksum, false);
    write_operator_fixture(bad_checksum_path, prefix, suffix, costs, {});
    std::vector<uint32_t> invalid_prefix = prefix;
    invalid_prefix[0] = invalid_prefix[1];
    write_operator_fixture(bad_permutation_path,
                           invalid_prefix,
                           suffix,
                           costs,
                           invalid_permutation_checksum);
    std::vector<uint32_t> invalid_order = prefix;
    std::swap(invalid_order[3], invalid_order[4]);
    write_operator_fixture(
        bad_order_path, invalid_order, suffix, costs, invalid_order_checksum);
    write_operator_fixture(bad_size_path, prefix, suffix, costs, valid_checksum);
    {
        std::ofstream output(bad_size_path, std::ios::binary | std::ios::app);
        output.put('\0');
    }
    std::vector<std::string> nested_texts;
    nested_texts.reserve(80);
    for (uint32_t idx = 0; idx < 80; ++idx) {
        char text[16];
        std::snprintf(text, sizeof(text), "prefix%03u", idx);
        nested_texts.emplace_back(text);
    }
    std::vector<uint32_t> nested_prefix(80);
    std::iota(nested_prefix.begin(), nested_prefix.end(), 0);
    const std::vector<uint32_t> nested_suffix{
        0, 10, 20, 30, 40, 50, 60, 70, 1, 11, 21, 31, 41, 51, 61, 71,
        2, 12, 22, 32, 42, 52, 62, 72, 3, 13, 23, 33, 43, 53, 63, 73,
        4, 14, 24, 34, 44, 54, 64, 74, 5, 15, 25, 35, 45, 55, 65, 75,
        6, 16, 26, 36, 46, 56, 66, 76, 7, 17, 27, 37, 47, 57, 67, 77,
        8, 18, 28, 38, 48, 58, 68, 78, 9, 19, 29, 39, 49, 59, 69, 79};
    std::vector<uint32_t> nested_costs(80);
    for (uint32_t idx = 0; idx < nested_costs.size(); ++idx) {
        nested_costs[idx] = 1000 - idx;
    }
    write_ngram_fixture(nested_ngram_path, nested_texts, nested_ngram_checksum);
    write_operator_fixture(nested_operator_path,
                           nested_prefix,
                           nested_suffix,
                           nested_costs,
                           nested_operator_checksum);

    const auto evaluate = [&](const std::string& pattern,
                              const std::string& mode,
                              float min_score,
                              uint32_t max_terms) {
        return evaluate_fuzzy_operator_rewrite_for_test(
            ngram_path, operator_path, pattern, mode, min_score, max_terms);
    };
    const auto evaluate_nested = [&](const std::string& pattern,
                                     const std::string& mode,
                                     float min_score,
                                     uint32_t max_terms) {
        return evaluate_fuzzy_operator_rewrite_for_test(
            ngram_path,
            operator_path,
            pattern,
            mode,
            min_score,
            max_terms,
            "direct_then_fallback");
    };
    const auto evaluate_literal = [&](const std::string& pattern,
                                      const std::string& mode,
                                      uint32_t max_terms) {
        return evaluate_fuzzy_operator_rewrite_for_test(
            ngram_path,
            operator_path,
            pattern,
            mode,
            1.0f,
            max_terms,
            "literal_only");
    };

    require_labels(evaluate("abc", "prefix", 0.25f, 1), {9}, false,
                   "prefix direct-only token-id ordering");
    require_labels(evaluate("eta", "suffix", 0.25f, 2), {3, 4}, false, "suffix");
    const FuzzyOperatorRewriteTestResult contains = evaluate("abcd", "contains", 0.25f, 2);
    require_labels(contains, {9, 10}, false, "contains");
    require(contains.direct_candidates == 4 && contains.direct_matches == 3,
            "contains literal verification counters mismatch");
    require_labels(evaluate("al", "prefix", 0.25f, 2), {8}, true,
                   "short prefix fallback");
    require_labels(evaluate("al", "suffix", 0.25f, 2), {8}, true,
                   "short suffix fallback");
    require_labels(evaluate("bet", "prefix", 0.10f, 3), {1, 3, 6}, true,
                   "direct-insufficient fallback");

    const FuzzyOperatorRewriteTestResult literal_prefix =
        evaluate_literal("al", "prefix", 32768);
    require(literal_prefix.operator_labels == std::vector<uint32_t>({0, 1, 2, 8}) &&
                literal_prefix.operator_relevance_ranks ==
                    std::vector<uint32_t>({3, 6, 4, 0}) &&
                !literal_prefix.used_fallback,
            "literal-only prefix labels or length-gap relevance ranks mismatch");
    const FuzzyOperatorRewriteTestResult literal_suffix =
        evaluate_literal("eta", "suffix", 32768);
    require(literal_suffix.operator_labels == std::vector<uint32_t>({3, 4, 5}) &&
                literal_suffix.operator_relevance_ranks ==
                    std::vector<uint32_t>({1, 2, 1}) &&
                !literal_suffix.used_fallback,
            "literal-only suffix labels or length-gap relevance ranks mismatch");
    const FuzzyOperatorRewriteTestResult literal_contains =
        evaluate_literal("abcd", "contains", 32768);
    const FuzzyOperatorRewriteTestResult legacy_contains_wide =
        evaluate("abcd", "contains", 0.25f, 32768);
    require(literal_contains.operator_labels == std::vector<uint32_t>({9, 10, 12}) &&
                literal_contains.operator_relevance_ranks ==
                    std::vector<uint32_t>({0, 1, 2}),
            "literal-only contains labels or length-gap relevance ranks mismatch");
    require(std::find(literal_contains.operator_labels.begin(),
                      literal_contains.operator_labels.end(),
                      11) == literal_contains.operator_labels.end() &&
                std::find(legacy_contains_wide.operator_labels.begin(),
                          legacy_contains_wide.operator_labels.end(),
                          11) != legacy_contains_wide.operator_labels.end(),
            "literal-only contains did not exclude the fixture's ngram-Jaccard false positive");
    require(literal_contains.direct_candidates == 4 &&
                literal_contains.direct_matches == 3 &&
                !literal_contains.used_fallback,
            "literal-only contains counters or fallback state mismatch");
    const FuzzyOperatorRewriteTestResult literal_short_contains =
        evaluate_literal("al", "contains", 32768);
    require(literal_short_contains.operator_labels ==
                std::vector<uint32_t>({0, 1, 2, 7, 8}) &&
                literal_short_contains.operator_relevance_ranks ==
                    std::vector<uint32_t>({3, 6, 4, 4, 0}) &&
                literal_short_contains.direct_candidates == texts.size(),
            "literal-only short contains did not scan and verify the full vocabulary");
    const FuzzyOperatorRewriteTestResult literal_cap =
        evaluate_literal("al", "prefix", 2);
    require(literal_cap.operator_labels == std::vector<uint32_t>({0, 1, 2, 8}) &&
                literal_cap.operator_relevance_ranks ==
                    std::vector<uint32_t>({3, 6, 4, 0}),
            "literal expansion applied the semantic cap before policy selection");
    for (const FuzzyOperatorRewriteTestResult* result : {
             &literal_prefix, &literal_suffix, &literal_contains,
             &literal_short_contains, &literal_cap}) {
        require_occurrence_metadata_aligned(*result, "literal-only rewrite");
        require(std::all_of(result->operator_source_tiers.begin(),
                            result->operator_source_tiers.end(),
                            [](uint8_t tier) {
                                return tier == static_cast<uint8_t>(
                                    QueryLabelSourceTier::Direct);
                            }),
                "literal-only source-tier metadata is not Direct aligned");
    }
    require_throws_with_message(
        [&] { (void)evaluate_literal("al", "prefix", 0); },
        "must be positive",
        "literal-only accepted a zero max_fuzzy_terms cap");
    require_throws_with_message(
        [&] { (void)evaluate_literal("al", "unsupported", 32768); },
        "supports only prefix, suffix, and contains",
        "literal-only accepted an unsupported fuzzy operator");

    const FuzzyOperatorRewriteTestResult nested_insufficient =
        evaluate_nested("bet", "prefix", 0.10f, 3);
    require(nested_insufficient.operator_labels == std::vector<uint32_t>({1, 3, 6}),
            "direct-then-fallback did not fill from baseline relevance order");
    require(nested_insufficient.used_fallback,
            "direct-insufficient nested fill did not report fallback");
    require_occurrence_metadata_aligned(nested_insufficient,
                                        "direct-insufficient nested fill");
    require(nested_insufficient.operator_source_tiers == std::vector<uint8_t>({
                static_cast<uint8_t>(QueryLabelSourceTier::Fallback),
                static_cast<uint8_t>(QueryLabelSourceTier::Direct),
                static_cast<uint8_t>(QueryLabelSourceTier::Direct)}) &&
                nested_insufficient.operator_relevance_ranks ==
                    std::vector<uint32_t>({1, 0, 0}),
            "metadata did not stay aligned after final label sorting");

    const FuzzyOperatorRewriteTestResult nested_cap2 =
        evaluate_nested("bet", "prefix", 0.10f, 2);
    require(std::includes(nested_insufficient.operator_labels.begin(),
                          nested_insufficient.operator_labels.end(),
                          nested_cap2.operator_labels.begin(),
                          nested_cap2.operator_labels.end()),
            "direct-insufficient nested cap2 labels are not a subset of cap3");

    const FuzzyManifestRewriteTestResult multi_predicate =
        evaluate_fuzzy_operator_manifest_row_for_test(
            ngram_path,
            operator_path,
            {"bet", "abc"},
            {"prefix", "contains"},
            0.10f,
            7,
            "literal_only");
    const FuzzyManifestRewriteTestResult multi_predicate_repeat =
        evaluate_fuzzy_operator_manifest_row_for_test(
            ngram_path,
            operator_path,
            {"bet", "abc"},
            {"prefix", "contains"},
            0.10f,
            7,
            "literal_only");
    require(multi_predicate.labels == multi_predicate_repeat.labels &&
                multi_predicate.source_tiers == multi_predicate_repeat.source_tiers &&
                multi_predicate.relevance_ranks == multi_predicate_repeat.relevance_ranks,
            "multi-predicate nested merge is not deterministic");
    require(multi_predicate.labels == std::vector<uint32_t>({3, 6, 9, 10, 11, 12}) &&
                multi_predicate.relevance_ranks ==
                    std::vector<uint32_t>({1, 4, 1, 2, 3, 3}) &&
                std::all_of(multi_predicate.source_tiers.begin(),
                            multi_predicate.source_tiers.end(),
                            [](uint8_t tier) {
                                return tier == static_cast<uint8_t>(
                                    QueryLabelSourceTier::Direct);
                            }),
            "multi-predicate literal expansion emitted a non-literal occurrence");

    // The move-only single-predicate fast path and the generic occurrence
    // merge must preserve the complete match set even with a small route cap.
    for (const std::string mode : {"prefix", "suffix", "contains"})
    for (const auto patterns : {std::vector<std::string>{mode == "suffix" ? "ta" : "bet"},
                               std::vector<std::string>{mode == "suffix" ? "ta" : "bet",
                                                        mode == "suffix" ? "bcd" : "abc",
                                                        mode == "suffix" ? "ta" : "bet"}}) {
        const std::vector<std::string> modes(patterns.size(), mode);
        configure_query_preparation({1, false});
        const auto reference = evaluate_fuzzy_operator_manifest_row_for_test(
            ngram_path, operator_path, patterns, modes, 0.10f, 1, "literal_only");
        configure_query_preparation({8, true});
        const auto actual = evaluate_fuzzy_operator_manifest_row_for_test(
            ngram_path, operator_path, patterns, modes, 0.10f, 1, "literal_only");
        require(reference.labels == actual.labels &&
                    reference.source_tiers == actual.source_tiers &&
                    reference.relevance_ranks == actual.relevance_ranks,
                "optimized manifest expansion changed complete labels or metadata");
    }
    configure_query_preparation({});

    for (const char* mode_value : {"prefix", "suffix", "contains"}) {
        const std::string mode(mode_value);
        const FuzzyOperatorRewriteTestResult short_pattern =
            evaluate_nested("al", mode, 0.25f, 2);
        require(short_pattern.used_fallback,
                "short " + mode + " pattern bypassed the baseline fallback");
        require_occurrence_metadata_aligned(short_pattern, "short " + mode);
        require(std::all_of(short_pattern.operator_source_tiers.begin(),
                            short_pattern.operator_source_tiers.end(),
                            [](uint8_t tier) {
                                return tier == static_cast<uint8_t>(
                                    QueryLabelSourceTier::Fallback);
                            }),
                "short " + mode + " pattern retained a direct source tier");
    }

    const FuzzyOperatorRewriteTestResult pure_fallback =
        evaluate_nested("abc", "unsupported", 0.0f, 3);
    require(pure_fallback.used_fallback,
            "unsupported operator mode did not use pure baseline fallback");
    require(std::all_of(pure_fallback.operator_source_tiers.begin(),
                        pure_fallback.operator_source_tiers.end(),
                        [](uint8_t tier) {
                            return tier == static_cast<uint8_t>(
                                QueryLabelSourceTier::Fallback);
                        }),
            "pure baseline fallback emitted a non-fallback source tier");

    const FuzzyOperatorRewriteTestResult cap32 =
        evaluate_fuzzy_operator_rewrite_for_test(nested_ngram_path,
                                                 nested_operator_path,
                                                 "prefix",
                                                 "prefix",
                                                 0.25f,
                                                 32,
                                                 "direct_then_fallback");
    const FuzzyOperatorRewriteTestResult cap64 =
        evaluate_fuzzy_operator_rewrite_for_test(nested_ngram_path,
                                                 nested_operator_path,
                                                 "prefix",
                                                 "prefix",
                                                 0.25f,
                                                 64,
                                                 "direct_then_fallback");
    require(cap32.operator_labels.size() == 32 && cap64.operator_labels.size() == 64,
            "nested cap fixture did not produce direct-sufficient cap32/cap64 rows");
    require(std::equal(cap32.operator_labels.begin(),
                       cap32.operator_labels.end(),
                       cap64.operator_labels.begin()),
            "cap32 direct labels are not a prefix of cap64");
    require(std::includes(cap64.operator_labels.begin(),
                          cap64.operator_labels.end(),
                          cap32.operator_labels.begin(),
                          cap32.operator_labels.end()),
            "cap32 direct labels are not a subset of cap64");
    require(!cap32.used_fallback && !cap64.used_fallback,
            "direct-sufficient cap fixture unexpectedly used fallback");
    require(std::all_of(cap64.operator_source_tiers.begin(),
                        cap64.operator_source_tiers.end(),
                        [](uint8_t tier) {
                            return tier == static_cast<uint8_t>(QueryLabelSourceTier::Direct);
                        }),
            "direct-sufficient nested fill emitted a fallback source tier");
    exercise_fuzzy_operator_rewrite_concurrently_for_test(
        ngram_path, operator_path, "abc", "prefix", 0.25f, 1, 48, 32);
    exercise_fuzzy_operator_rewrite_concurrently_for_test(
        ngram_path, operator_path, "bet", "prefix", 0.10f, 3, 48, 16);
    exercise_fuzzy_operator_rewrite_concurrently_for_test(
        ngram_path,
        operator_path,
        "bet",
        "prefix",
        0.10f,
        3,
        48,
        16,
        "direct_then_fallback");

    const std::string config_path = temporary_path(".config.json");
    {
        std::ofstream output(config_path);
        output << "{\"query_manifest_operator_index_path\":\"" << operator_path
               << "\",\"query_manifest_operator_fill_mode\":\"legacy_full_fallback\""
               << ",\"query_manifest_operator_parity_check\":true}";
    }
    const go_fann::AppConfigValues config = go_fann::load_config_file(config_path);
    require(config.query_manifest_operator_index_path == operator_path,
            "operator config path was not loaded");
    require(config.query_manifest_operator_parity_check.value_or(false),
            "operator parity config was not loaded");
    require(config.query_manifest_operator_fill_mode == "legacy_full_fallback",
            "operator fill mode config was not loaded");
    {
        std::ofstream output(nested_config_path);
        output << R"({"runtime":{"query_manifest_operator_fill_mode":"direct_then_fallback"}})";
    }
    require(go_fann::load_config_file(nested_config_path)
                .query_manifest_operator_fill_mode == "direct_then_fallback",
            "nested operator fill mode config was not loaded");
    {
        std::ofstream output(literal_config_path);
        output << R"({"runtime":{"query_manifest_operator_fill_mode":"literal_only","query_manifest_max_fuzzy_terms":32768,"query_manifest_operator_parity_check":false}})";
    }
    const go_fann::AppConfigValues literal_config =
        go_fann::load_config_file(literal_config_path);
    require(literal_config.query_manifest_operator_fill_mode == "literal_only" &&
                literal_config.query_manifest_max_fuzzy_terms.value_or(0) == 32768 &&
                !literal_config.query_manifest_operator_parity_check.value_or(true),
            "literal-only fill/cap/parity config was not loaded");
    require(is_valid_query_manifest_operator_fill_mode("legacy_full_fallback") &&
                is_valid_query_manifest_operator_fill_mode("direct_then_fallback") &&
                is_valid_query_manifest_operator_fill_mode("literal_only") &&
                !is_valid_query_manifest_operator_fill_mode("unknown"),
            "operator fill mode validation accepted the wrong value set");
    require_throws_with_message(
        [] {
            validate_query_manifest_operator_rewrite_options(
                "direct_then_fallback", true);
        },
        "conflicts",
        "nested fill plus legacy parity_check was accepted");
    require_throws_with_message(
        [] {
            validate_query_manifest_operator_rewrite_options(
                "literal_only", true);
        },
        "conflicts",
        "literal-only plus parity_check was accepted");
    validate_query_manifest_fuzzy_operator_fill_mode("literal_only");
    require_throws_with_message(
        [] {
            validate_query_manifest_fuzzy_operator_fill_mode(
                "direct_then_fallback");
        },
        "fallback tokens do not satisfy",
        "manifest fuzzy accepted fallback-token expansion");

    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, bad_magic_path, "abc", "prefix", 0.25f, 1); },
        "magic", "bad magic was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, bad_checksum_path, "abc", "prefix", 0.25f, 1); },
        "checksum", "bad checksum was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, bad_permutation_path, "abc", "prefix", 0.25f, 1); },
        "permutation", "bad permutation was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, bad_order_path, "abc", "prefix", 0.25f, 1); },
        "not sorted", "bad lexical order was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, bad_size_path, "abc", "prefix", 0.25f, 1); },
        "size mismatch", "operator trailing bytes were accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  bad_ngram_checksum_path, operator_path, "abc", "prefix", 0.25f, 1); },
        "ngram payload checksum", "bad ngram checksum was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  bad_ngram_size_path, operator_path, "abc", "prefix", 0.25f, 1); },
        "ngram index size mismatch", "ngram trailing bytes were accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  bad_ngram_offsets_path, operator_path, "abc", "prefix", 0.25f, 1); },
        "token text offsets", "bad ngram offsets were accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  short_ngram_path, operator_path, "abc", "prefix", 0.25f, 1); },
        "does not match", "operator/ngram token count mismatch was accepted");
    require_throws_with_message(
        [&] { (void)evaluate_fuzzy_operator_rewrite_for_test(
                  ngram_path, operator_path + ".missing", "abc", "prefix", 0.25f, 1); },
        "explicitly configured", "missing configured operator sidecar was accepted");

    const char* production_ngram = std::getenv("GO_FANN_TEST_TOKEN_NGRAM_INDEX");
    const char* production_operator = std::getenv("GO_FANN_TEST_TOKEN_OPERATOR_INDEX");
    if (production_ngram != nullptr || production_operator != nullptr) {
        require(production_ngram != nullptr && production_operator != nullptr,
                "both production artifact test paths must be configured");
        const FuzzyOperatorRewriteTestResult production_result =
            evaluate_fuzzy_operator_rewrite_for_test(
                production_ngram, production_operator, "the", "contains", 0.25f, 32);
        require(production_result.operator_labels == production_result.baseline_labels,
                "production artifact smoke parity mismatch");
        std::puts("production operator artifact smoke passed");
    }

    for (const std::string& path : {ngram_path,
                                    short_ngram_path,
                                    bad_ngram_checksum_path,
                                    bad_ngram_size_path,
                                    bad_ngram_offsets_path,
                                    operator_path,
                                    bad_magic_path,
                                    bad_checksum_path,
                                    bad_permutation_path,
                                    bad_order_path,
                                    bad_size_path,
                                    nested_ngram_path,
                                    nested_operator_path,
                                    nested_config_path,
                                    literal_config_path,
                                    config_path}) {
        std::remove(path.c_str());
    }
    std::puts("operator fuzzy rewrite tests passed");
    return 0;
}
