"""
blocking.py - Candidate generation / blocking strategies.

Implements 6 inverted-index blocking strategies (A-F) plus TF-IDF (G),
with union blocking (H = A+B+C+D+E+F).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalize import (
    get_name_tokens, get_addr_tokens, get_bigrams, extract_numbers
)

log = logging.getLogger(__name__)


class BlockingIndex:
    """
    Multi-strategy inverted-index blocking.

    Strategies:
      A - Exact normalized name
      B - Name token inverted index
      C - Name character bigram index
      D - Address token index
      E - Country + 4-char name prefix
      F - Country + address numerics
      G - TF-IDF top-K retrieval (optional, expensive)
    """

    def __init__(self):
        self.name_exact: Dict[str, Set[str]]  = defaultdict(set)
        self.name_token: Dict[str, Set[str]]  = defaultdict(set)
        self.name_bigram: Dict[str, Set[str]] = defaultdict(set)
        self.addr_token: Dict[str, Set[str]]  = defaultdict(set)
        self.cntry_pfx: Dict[str, Set[str]]   = defaultdict(set)
        self.cntry_num: Dict[str, Set[str]]   = defaultdict(set)
        # TF-IDF
        self.tfidf_ids: List[str] = []
        self.tfidf_texts: List[str] = []
        self.tfidf_vec: Optional[TfidfVectorizer] = None
        self.tfidf_mat = None

    def add(self, eid: str, norm_name: str, norm_addr: str,
            norm_country: str) -> None:
        """Index one S2/S3 record."""
        # A
        if norm_name:
            self.name_exact[norm_name].add(eid)
        # B
        for tok in get_name_tokens(norm_name):
            self.name_token[tok].add(eid)
        # C - first 30 chars to bound index size
        for bg in get_bigrams(norm_name[:30]):
            self.name_bigram[bg].add(eid)
        # D
        for tok in get_addr_tokens(norm_addr):
            self.addr_token[tok].add(eid)
        # E
        if norm_name and norm_country:
            pfx = norm_name[:4].strip()
            if pfx:
                self.cntry_pfx[norm_country + '|' + pfx].add(eid)
        # F
        nums = extract_numbers(norm_addr)
        if nums and norm_country:
            for n in nums[:3]:
                self.cntry_num[norm_country + '|' + n].add(eid)
        # Store for TF-IDF
        self.tfidf_ids.append(eid)
        self.tfidf_texts.append((norm_name + ' ' + norm_addr).strip())

    def build_tfidf(self, max_features: int = 80000) -> None:
        """Build TF-IDF character-ngram matrix (call once after all adds)."""
        log.info(f"Building TF-IDF on {len(self.tfidf_texts):,} records...")
        self.tfidf_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 3),
            max_features=max_features, min_df=2, sublinear_tf=True
        )
        self.tfidf_mat = self.tfidf_vec.fit_transform(self.tfidf_texts)
        log.info(f"TF-IDF matrix: {self.tfidf_mat.shape}")

    def _strategy(self, s: str, norm_name: str, norm_addr: str,
                  norm_country: str) -> Set[str]:
        """Single-strategy retrieval."""
        if s == 'A':
            return self.name_exact.get(norm_name, set()).copy() if norm_name else set()
        if s == 'B':
            out = set()
            for tok in get_name_tokens(norm_name):
                out |= self.name_token.get(tok, set())
            return out
        if s == 'C':
            out = set()
            for bg in get_bigrams(norm_name[:30]):
                out |= self.name_bigram.get(bg, set())
            return out
        if s == 'D':
            out = set()
            for tok in get_addr_tokens(norm_addr):
                out |= self.addr_token.get(tok, set())
            return out
        if s == 'E':
            if norm_name and norm_country:
                pfx = norm_name[:4].strip()
                if pfx:
                    return self.cntry_pfx.get(norm_country + '|' + pfx, set()).copy()
            return set()
        if s == 'F':
            out = set()
            nums = extract_numbers(norm_addr)
            if nums and norm_country:
                for n in nums[:3]:
                    out |= self.cntry_num.get(norm_country + '|' + n, set())
            return out
        if s == 'G':
            return self._tfidf_cands(norm_name, norm_addr, top_k=15)
        return set()

    def _tfidf_cands(self, norm_name: str, norm_addr: str,
                     top_k: int = 15) -> Set[str]:
        if self.tfidf_vec is None:
            return set()
        query = (norm_name + ' ' + norm_addr).strip()
        if not query:
            return set()
        q = self.tfidf_vec.transform([query])
        sims = cosine_similarity(q, self.tfidf_mat).flatten()
        idx = np.argsort(sims)[-top_k:][::-1]
        return {self.tfidf_ids[i] for i in idx if sims[i] > 0.0}

    def get_candidates(self, norm_name: str, norm_addr: str,
                       norm_country: str,
                       strategies: List[str],
                       max_cands: int = 300) -> Set[str]:
        """Union candidates across strategies, capped at max_cands."""
        out = set()
        for s in strategies:
            out |= self._strategy(s, norm_name, norm_addr, norm_country)
            if len(out) > max_cands:
                break
        return out


def build_index(df: pd.DataFrame) -> BlockingIndex:
    """Build a BlockingIndex from a dataframe with norm_* columns."""
    idx = BlockingIndex()
    for _, row in df.iterrows():
        idx.add(row['entity_id'], row['norm_name'],
                row['norm_addr'], row['norm_country'])
    return idx


def evaluate_strategy(s1_sample: pd.DataFrame, idx: BlockingIndex,
                      gt: Dict[str, List[str]],
                      strategies: List[str]) -> dict:
    """
    Evaluate a blocking strategy on a sample of S1 entities.
    Returns recall, avg/median/max candidates, reduction ratio.
    """
    found = 0
    total_true = 0
    total_cands = 0
    counts = []
    n_s23 = len(idx.tfidf_ids)

    for _, row in s1_sample.iterrows():
        eid = row['entity_id']
        true_set = set(gt.get(eid, []))
        cands = idx.get_candidates(row['norm_name'], row['norm_addr'],
                                   row['norm_country'], strategies)
        found += len(true_set & cands)
        total_true += len(true_set)
        total_cands += len(cands)
        counts.append(len(cands))

    n = len(s1_sample)
    recall = found / max(total_true, 1)
    avg = total_cands / max(n, 1)
    reduction = 1.0 - total_cands / max(n * n_s23, 1)

    return {
        'recall': recall,
        'avg_candidates': avg,
        'median_candidates': float(np.median(counts)),
        'max_candidates': int(max(counts)) if counts else 0,
        'total_candidates': total_cands,
        'reduction_ratio': reduction,
    }
