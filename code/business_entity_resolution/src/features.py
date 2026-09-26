"""
features.py - Feature engineering for candidate pair scoring.

30 features covering name similarity, address similarity,
country agreement, and combined signals.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

from normalize import (
    get_name_tokens, get_addr_tokens, get_bigrams, get_trigrams,
    extract_numbers, normalize_country
)

FEATURE_NAMES = [
    # Name features (0-12)
    'name_exact',          # 0
    'name_lev',            # 1 Levenshtein ratio
    'name_wRatio',         # 2 Weighted ratio (Jaro-Winkler variant)
    'name_token_set',      # 3 Token set ratio
    'name_token_sort',     # 4 Token sort ratio
    'name_jaccard',        # 5 Jaccard on tokens
    'name_overlap',        # 6 Overlap coefficient on tokens
    'name_bigram_jac',     # 7 Bigram Jaccard
    'name_trigram_jac',    # 8 Trigram Jaccard
    'name_len_diff',       # 9 Normalized length difference
    'name_common_tok',     # 10 Raw common token count
    'name_ntok_s1',        # 11
    'name_ntok_s2',        # 12
    # Address features (13-22)
    'addr_exact',          # 13
    'addr_lev',            # 14
    'addr_token_set',      # 15
    'addr_jaccard',        # 16
    'addr_overlap',        # 17
    'addr_bigram_jac',     # 18
    'addr_num_jac',        # 19 Numeric token Jaccard
    'addr_len_diff',       # 20
    'addr_both_empty',     # 21
    'addr_one_empty',      # 22
    # Country features (23-24)
    'country_exact',       # 23
    'country_both_have',   # 24
    # Combined (25-29)
    'name_x_addr',         # 25
    'weighted_combined',   # 26
    'max_name_sim',        # 27
    'has_common_name_tok', # 28
    'has_common_addr_tok', # 29
]

N_FEATURES = len(FEATURE_NAMES)


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u > 0 else 0.0


def _overlap(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def compute_features(
    s1_norm_name: str, s1_norm_addr: str, s1_country: str,
    s2_norm_name: str, s2_norm_addr: str, s2_country: str,
) -> List[float]:
    """
    Compute 30-dimensional feature vector for a candidate pair.

    Parameters
    ----------
    s1_norm_name, s1_norm_addr, s1_country : Source 1 normalized fields
    s2_norm_name, s2_norm_addr, s2_country : Source 2/3 normalized fields

    Returns
    -------
    list of float, length == N_FEATURES
    """
    f = []

    # ─── NAME ──────────────────────────────────────────────
    f.append(float(s1_norm_name == s2_norm_name and s1_norm_name != ''))  # 0
    f.append(fuzz.ratio(s1_norm_name, s2_norm_name) / 100.0)              # 1
    f.append(fuzz.WRatio(s1_norm_name, s2_norm_name) / 100.0)             # 2
    f.append(fuzz.token_set_ratio(s1_norm_name, s2_norm_name) / 100.0)    # 3
    f.append(fuzz.token_sort_ratio(s1_norm_name, s2_norm_name) / 100.0)   # 4

    t1n = get_name_tokens(s1_norm_name)
    t2n = get_name_tokens(s2_norm_name)
    common_name_tok = len(t1n & t2n)

    f.append(_jaccard(t1n, t2n))       # 5
    f.append(_overlap(t1n, t2n))       # 6
    f.append(_jaccard(get_bigrams(s1_norm_name), get_bigrams(s2_norm_name)))    # 7
    f.append(_jaccard(get_trigrams(s1_norm_name), get_trigrams(s2_norm_name)))  # 8

    l1n, l2n = len(s1_norm_name), len(s2_norm_name)
    f.append(abs(l1n - l2n) / max(l1n + l2n, 1))  # 9
    f.append(float(common_name_tok))               # 10
    f.append(float(len(t1n)))                      # 11
    f.append(float(len(t2n)))                      # 12

    # ─── ADDRESS ───────────────────────────────────────────
    f.append(float(s1_norm_addr == s2_norm_addr and s1_norm_addr != ''))  # 13
    f.append(fuzz.ratio(s1_norm_addr, s2_norm_addr) / 100.0)              # 14
    f.append(fuzz.token_set_ratio(s1_norm_addr, s2_norm_addr) / 100.0)    # 15

    t1a = get_addr_tokens(s1_norm_addr)
    t2a = get_addr_tokens(s2_norm_addr)
    common_addr_tok = len(t1a & t2a)

    f.append(_jaccard(t1a, t2a))   # 16
    f.append(_overlap(t1a, t2a))   # 17
    f.append(_jaccard(get_bigrams(s1_norm_addr[:50]),
                      get_bigrams(s2_norm_addr[:50])))  # 18

    n1 = set(extract_numbers(s1_norm_addr))
    n2 = set(extract_numbers(s2_norm_addr))
    f.append(_jaccard(n1, n2))     # 19

    l1a, l2a = len(s1_norm_addr), len(s2_norm_addr)
    f.append(abs(l1a - l2a) / max(l1a + l2a, 1))  # 20
    f.append(float(l1a == 0 and l2a == 0))         # 21
    f.append(float((l1a == 0) != (l2a == 0)))      # 22

    # ─── COUNTRY ───────────────────────────────────────────
    c1 = normalize_country(s1_country)
    c2 = normalize_country(s2_country)
    f.append(float(c1 == c2))          # 23
    f.append(float(bool(c1 and c2)))   # 24

    # ─── COMBINED ──────────────────────────────────────────
    name_sim = f[3]   # token_set_ratio
    addr_sim = f[15]  # addr token_set_ratio
    ctry_sim = f[23]

    f.append(name_sim * addr_sim)                              # 25
    f.append(0.6 * name_sim + 0.3 * addr_sim + 0.1 * ctry_sim)  # 26
    f.append(max(f[1], f[2], f[3], f[4]))                     # 27
    f.append(float(common_name_tok > 0))                      # 28
    f.append(float(common_addr_tok > 0))                      # 29

    return f


# Lookup type: entity_id -> (norm_name, norm_addr, country)
Lookup = Dict[str, Tuple[str, str, str]]


def build_lookup(df) -> Lookup:
    """Build entity_id -> (norm_name, norm_addr, country) lookup dict."""
    return {
        row['entity_id']: (row['norm_name'], row['norm_addr'], row['country'])
        for _, row in df.iterrows()
    }


def build_feature_matrix(pairs: List[Tuple[str, str]],
                          s1_lookup: Lookup,
                          s23_lookup: Lookup) -> np.ndarray:
    """
    Build feature matrix for a list of (s1_id, cand_id) pairs.

    Returns ndarray of shape (len(pairs), N_FEATURES).
    """
    rows = []
    zero = [0.0] * N_FEATURES
    for s1_id, cand_id in pairs:
        s1r  = s1_lookup.get(s1_id)
        candr = s23_lookup.get(cand_id)
        if s1r is None or candr is None:
            rows.append(zero)
            continue
        rows.append(compute_features(
            s1r[0], s1r[1], s1r[2],
            candr[0], candr[1], candr[2]
        ))
    return np.array(rows, dtype=np.float32)
