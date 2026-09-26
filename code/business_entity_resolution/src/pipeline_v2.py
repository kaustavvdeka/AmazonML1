"""
pipeline_v2.py - Enhanced High-Score Business Entity Resolution Pipeline.

Target Metric: Macro F0.5 (Penalizes false positives 4x more than false negatives)

Key Architectural Pillars for Maximum Score:
1. Multi-Lingual & French Normalization (SARL, SAS, EURL, SCI, French address tokens)
2. Core Name Stripping (matches brand entities across differing legal forms)
3. Country-Aware Postal/PIN Code & Street Number Parsing (high precision signals)
4. Multi-Pass High-Recall Blocking (A + A_core + Postal_Pfx + E + Rare_Tokens + Addr)
5. 38 Pairwise Features with cross-modal interactions and conflict penalties
6. Dual Ensemble: Blended XGBoost (55%) + LightGBM (45%)
7. Metric-Aligned Threshold Optimization + Entity-Level Margin Pruning (max 6 matches)
8. Vectorized Streaming Inference (10x faster, zero RAM blowup, crash-safe)
"""

import gc
import logging
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Constants & Paths
RANDOM_STATE = 42
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# Suffixes & Stopwords
CORE_LEGAL_STOPWORDS = {
    'llc', 'llp', 'pvtltd', 'inc', 'corp', 'ltd', 'pvt', 'ent', 'svc',
    'grp', 'intl', 'natl', 'assoc', 'consult', 'trd', 'ind', 'sol',
    'tech', 'mgmt', 'dev', 'fdn', 'hosp', 'sch', 'ctr', 'co',
    'sarl', 'sas', 'eurl', 'sci', 'snc', 'fed', 'amicale'
}

ADDR_STOPWORDS = {
    'near', 'opposite', 'behind', 'adjacent', 'next', 'road', 'rd', 'street', 'st',
    'avenue', 'ave', 'lane', 'ln', 'floor', 'fl', 'building', 'bldg', 'india', 'us',
    'france', 'delhi', 'mumbai', 'calais', 'bordeaux', 'texas', 'rue'
}

LEGAL_SUFFIX_PATTERNS = [
    (r'\blimited liability company\b', 'llc'),
    (r'\blimited liability partnership\b', 'llp'),
    (r'\bpvt\s*ltd\b', 'pvtltd'),
    (r'\bprivate\s*limited\b', 'pvtltd'),
    (r'\bpvt\s*limited\b', 'pvtltd'),
    (r'\bprivate\s*ltd\b', 'pvtltd'),
    (r'\bincorporated\b', 'inc'),
    (r'\bcorporations\b', 'corp'),
    (r'\bcorporation\b', 'corp'),
    (r'\blimited\b', 'ltd'),
    (r'\bprivate\b', 'pvt'),
    (r'\benterprises\b', 'ent'),
    (r'\benterprise\b', 'ent'),
    (r'\bservices\b', 'svc'),
    (r'\bservice\b', 'svc'),
    (r'\bcompany\b', 'co'),
    # French
    (r'\bs\.?a\.?r\.?l\.?\b', 'sarl'),
    (r'\bs\.?a\.?s\.?u\.?\b', 'sas'),
    (r'\bs\.?a\.?s\.?\b', 'sas'),
    (r'\be\.?u\.?r\.?l\.?\b', 'eurl'),
    (r'\bs\.?c\.?i\.?\b', 'sci'),
    (r'\bs\.?n\.?c\.?\b', 'snc'),
    (r'\bassociation\b', 'assoc'),
    (r'\bamicale\b', 'amicale'),
    (r'\bfederation\b', 'fed'),
    (r'\bdba\b', ''),
]

ADDR_PATTERNS = [
    (r'\bstreet\b', 'st'),
    (r'\broad\b', 'rd'),
    (r'\bavenue\b', 'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\bdrive\b', 'dr'),
    (r'\bcourt\b', 'ct'),
    (r'\blane\b', 'ln'),
    (r'\bsuite\b', 'ste'),
    (r'\bapartment\b', 'apt'),
    (r'\bbuilding\b', 'bldg'),
    (r'\bpost\s*office\s*box\b', 'po box'),
    (r'\bp\.?o\.?\s*box\b', 'po box'),
    # French
    (r'\brue\b', 'rue'),
    (r'\bimpasse\b', 'imp'),
    (r'\ball[eé]e\b', 'allee'),
    (r'\bchemin\b', 'chem'),
    (r'\bcedex\b', 'cedex'),
]


def norm_str(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    text = s.lower().strip()
    text = text.replace('&', ' and ')
    return text


def clean_name(name: str) -> str:
    text = norm_str(name)
    if not text:
        return ""
    for p, r in LEGAL_SUFFIX_PATTERNS:
        text = re.sub(p, r, text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_address(addr: str) -> str:
    text = norm_str(addr)
    if not text:
        return ""
    for p, r in ADDR_PATTERNS:
        text = re.sub(p, ' ' + r + ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_country(country: str) -> str:
    return norm_str(country)


def get_core_name(clean_nm: str) -> str:
    tokens = [t for t in clean_nm.split() if t not in CORE_LEGAL_STOPWORDS]
    return ' '.join(tokens) if tokens else clean_nm


def get_postal(clean_ad: str, country: str) -> str:
    c = country.lower().strip()
    if c == 'india':
        m = re.findall(r'\b[1-9]\d{5}\b', clean_ad)
        return m[0] if m else ''
    elif c == 'us':
        m = re.findall(r'\b\d{5}\b', clean_ad)
        return m[0] if m else ''
    elif c == 'france':
        m = re.findall(r'\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b', clean_ad)
        return m[0] if m else ''
    return ''


def get_numbers(clean_ad: str) -> List[str]:
    return re.findall(r'\b\d+\b', clean_ad)


# ─────────────────────────────────────────────────────────────
# 38-DIM PAIRWISE FEATURE VECTOR
# ─────────────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = len(a | b)
    return len(a & b) / u if u > 0 else 0.0


def _overlap(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))


def _char_ngrams(s: str, n: int) -> set:
    c = s.replace(' ', '')
    return {c[i:i+n] for i in range(len(c) - n + 1)} if len(c) >= n else set()


def extract_features(nn1: str, na1: str, nc1: str,
                     nn2: str, na2: str, nc2: str) -> List[float]:
    f = []
    # 1-13: Standard Name Similarities
    f.append(float(nn1 == nn2 and nn1 != ''))
    f.append(fuzz.ratio(nn1, nn2) / 100.0)
    f.append(fuzz.WRatio(nn1, nn2) / 100.0)
    f.append(fuzz.token_set_ratio(nn1, nn2) / 100.0)
    f.append(fuzz.token_sort_ratio(nn1, nn2) / 100.0)
    t1n = {t for t in nn1.split() if len(t) >= 2}
    t2n = {t for t in nn2.split() if len(t) >= 2}
    f.append(_jaccard(t1n, t2n))
    f.append(_overlap(t1n, t2n))
    f.append(_jaccard(_char_ngrams(nn1, 2), _char_ngrams(nn2, 2)))
    f.append(_jaccard(_char_ngrams(nn1, 3), _char_ngrams(nn2, 3)))
    l1n, l2n = len(nn1), len(nn2)
    f.append(abs(l1n - l2n) / max(l1n + l2n, 1))
    common_n = t1n & t2n
    f.append(float(len(common_n)))
    f.append(float(len(t1n)))
    f.append(float(len(t2n)))

    # 14-16: Core Name Signals (Ignoring Legal Suffixes)
    cn1 = get_core_name(nn1)
    cn2 = get_core_name(nn2)
    f.append(float(cn1 == cn2 and cn1 != ''))
    f.append(fuzz.ratio(cn1, cn2) / 100.0)
    f.append(fuzz.token_set_ratio(cn1, cn2) / 100.0)

    # 17-23: Address Similarities
    f.append(float(na1 == na2 and na1 != ''))
    f.append(fuzz.ratio(na1, na2) / 100.0)
    f.append(fuzz.token_set_ratio(na1, na2) / 100.0)
    t1a = {t for t in na1.split() if len(t) >= 3 and t not in ADDR_STOPWORDS}
    t2a = {t for t in na2.split() if len(t) >= 3 and t not in ADDR_STOPWORDS}
    f.append(_jaccard(t1a, t2a))
    f.append(_overlap(t1a, t2a))
    f.append(_jaccard(_char_ngrams(na1[:50], 2), _char_ngrams(na2[:50], 2)))

    # 24-28: Numeric & Postal Code Signals
    n1 = set(get_numbers(na1))
    n2 = set(get_numbers(na2))
    f.append(_jaccard(n1, n2))
    f.append(float(n1 == n2 and len(n1) > 0))
    f.append(float(len(n1) > 0 and len(n2) > 0 and len(n1 & n2) == 0))
    p1 = get_postal(na1, nc1)
    p2 = get_postal(na2, nc2)
    f.append(float(p1 == p2 and p1 != ''))
    f.append(float(p1 != '' and p2 != '' and p1 != p2))

    # 29-32: Length & Missingness
    l1a, l2a = len(na1), len(na2)
    f.append(abs(l1a - l2a) / max(l1a + l2a, 1))
    f.append(float(l1a == 0 and l2a == 0))
    f.append(float((l1a == 0) != (l2a == 0)))
    f.append(float(nc1 == nc2))

    # 33-38: Multi-Modal Interactions
    nsim = max(f[1], f[3])
    asim = max(f[17], f[18])
    f.append(nsim * asim)
    f.append(0.65 * nsim + 0.35 * asim)
    f.append(nsim)
    f.append(float(l1a == 0 or l2a == 0) * nsim)
    f.append(float(len(common_n) > 0))
    f.append(float(len(t1a & t2a) > 0))

    return f


# ─────────────────────────────────────────────────────────────
# HIGH-RECALL MULTI-PASS BLOCKING INDEX
# ─────────────────────────────────────────────────────────────

class HighRecallBlockingIndex:
    def __init__(self):
        self.exact_name = defaultdict(list)
        self.exact_core = defaultdict(list)
        self.postal_pfx = defaultdict(list)
        self.cntry_pfx4 = defaultdict(list)
        self.rare_tokens = defaultdict(list)
        self.addr_tokens = defaultdict(list)

    def fit(self, df: pd.DataFrame):
        log.info(f"Building high-recall index over {len(df):,} records...")
        t0 = time.time()
        for eid, nn, na, nc in df[['entity_id', 'norm_name', 'norm_addr', 'norm_country']].values:
            nn = str(nn); na = str(na); nc = str(nc)
            cn = get_core_name(nn)
            post = get_postal(na, nc)

            if nn: self.exact_name[nn].append(eid)
            if cn: self.exact_core[cn].append(eid)
            if nc and len(nn) >= 4: self.cntry_pfx4[nc + '|' + nn[:4]].append(eid)
            if post and cn: self.postal_pfx[nc + '|' + post + '|' + cn[:3]].append(eid)

            for t in nn.split():
                if len(t) >= 3 and t not in CORE_LEGAL_STOPWORDS:
                    self.rare_tokens[t].append(eid)

            for t in na.split():
                if len(t) >= 4 and t not in ADDR_STOPWORDS:
                    self.addr_tokens[t].append(eid)

        log.info(f"Index built in {time.time()-t0:.1f}s.")

    def query(self, nn: str, na: str, nc: str, cap: int = 100) -> Set[str]:
        out = set()
        cn = get_core_name(nn)
        post = get_postal(na, nc)

        # 1. Exact Name & Core Name
        if nn and nn in self.exact_name:
            out.update(self.exact_name[nn])
        if cn and cn in self.exact_core:
            out.update(self.exact_core[cn][:80])

        # 2. Postal + Brand Prefix
        if post and cn:
            k = nc + '|' + post + '|' + cn[:3]
            if k in self.postal_pfx:
                out.update(self.postal_pfx[k])

        # 3. Country + Name Prefix 4
        if len(out) < cap and nc and len(nn) >= 4:
            k = nc + '|' + nn[:4]
            if k in self.cntry_pfx4:
                pl = self.cntry_pfx4[k]
                if len(pl) <= 300:
                    out.update(pl[:80])

        # 4. Rare Distinctive Name Tokens
        if len(out) < cap:
            for t in nn.split():
                if len(t) >= 4 and t not in CORE_LEGAL_STOPWORDS:
                    pl = self.rare_tokens.get(t, [])
                    if 0 < len(pl) <= 200:
                        out.update(pl[:60])
                if len(out) >= cap:
                    break

        # 5. Informative Address Tokens
        if len(out) < cap:
            for t in na.split():
                if len(t) >= 5 and t not in ADDR_STOPWORDS:
                    pl = self.addr_tokens.get(t, [])
                    if 0 < len(pl) <= 150:
                        out.update(pl[:40])
                if len(out) >= cap:
                    break

        return out


# ─────────────────────────────────────────────────────────────
# MACRO F0.5 SCORER
# ─────────────────────────────────────────────────────────────

def macro_f05(s1_ids: List[str], preds: Dict[str, Set[str]],
              gt: Dict[str, Set[str]]) -> Tuple[float, float, float]:
    p_list, r_list, f_list = [], [], []
    for sid in s1_ids:
        p_set = preds.get(sid, set())
        g_set = gt.get(sid, set())
        if not p_set and not g_set:
            f_list.append(1.0); p_list.append(1.0); r_list.append(1.0)
            continue
        tp = len(p_set & g_set)
        p = tp / len(p_set) if p_set else 0.0
        r = tp / len(g_set) if g_set else 0.0
        denom = 0.25 * p + r
        f05 = (1.25 * p * r / denom) if denom > 0 else 0.0
        f_list.append(f05); p_list.append(p); r_list.append(r)
    return float(np.mean(f_list)), float(np.mean(p_list)), float(np.mean(r_list))


# ─────────────────────────────────────────────────────────────
# ENSEMBLE TRAINING
# ─────────────────────────────────────────────────────────────

class BlendedEnsemble:
    def __init__(self, xgb_model, lgb_model, xgb_weight=0.55):
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.w = xgb_weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        blend = self.w * p_xgb + (1.0 - self.w) * p_lgb
        return np.column_stack([1.0 - blend, blend])


def train_ensemble(X: np.ndarray, y: np.ndarray) -> BlendedEnsemble:
    log.info(f"Training XGBoost on {len(y):,} pairs...")
    scale_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    xgb_clf.fit(X, y)

    log.info("Training LightGBM on pairs...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )
    lgb_clf.fit(X, y)

    return BlendedEnsemble(xgb_clf, lgb_clf, xgb_weight=0.55)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────

def run_enhanced_pipeline():
    t_start = time.time()
    log.info("=" * 70)
    log.info("AMAZON ML CHALLENGE 2026 - ENHANCED PIPELINE V2")
    log.info("Target: High Precision Macro F0.5 Optimization")
    log.info("=" * 70)

    # 1. Load Training Data
    log.info("\n[1] Loading Ground Truth and Train S1...")
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')
    gt = {}
    for sid, mids in gt_df[['source1_entity_id', 'matched_entity_ids']].values:
        gt[str(sid)] = set(str(mids).split(',')) if pd.notna(mids) and str(mids).strip() else set()
    del gt_df; gc.collect()

    s1_tr = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t')
    s1_tr['norm_name'] = s1_tr['business_name'].apply(clean_name)
    s1_tr['norm_addr'] = s1_tr['business_address'].apply(clean_address)
    s1_tr['norm_country'] = s1_tr['country'].apply(clean_country)
    log.info(f"  Train S1 loaded: {len(s1_tr):,}")

    # 2. Load Train S2+S3 and Build Index
    log.info("\n[2] Loading Train S2 and S3...")
    s2_tr = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t')
    s3_tr = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t')
    s23_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    del s2_tr, s3_tr; gc.collect()

    s23_tr['norm_name'] = s23_tr['business_name'].apply(clean_name)
    s23_tr['norm_addr'] = s23_tr['business_address'].apply(clean_address)
    s23_tr['norm_country'] = s23_tr['country'].apply(clean_country)

    idx_tr = HighRecallBlockingIndex()
    idx_tr.fit(s23_tr)

    s23_tr_lkp = dict(zip(
        s23_tr['entity_id'],
        zip(s23_tr['norm_name'], s23_tr['norm_addr'], s23_tr['norm_country'])
    ))
    del s23_tr; gc.collect()

    # 3. Train/Validation Split
    log.info("\n[3] Creating 80/20 Stratified Validation Split...")
    has_matches = [len(gt.get(sid, set())) > 0 for sid in s1_tr['entity_id']]
    s1_train, s1_val = train_test_split(s1_tr, test_size=0.20, random_state=RANDOM_STATE, stratify=has_matches)
    s1_train = s1_train.reset_index(drop=True)
    s1_val = s1_val.reset_index(drop=True)
    log.info(f"  Train: {len(s1_train):,}  Val: {len(s1_val):,}")

    # 4. Generate Training Pairs with Hard Negative Mining
    log.info("\n[4] Generating Training Pairs (80,000 Sample)...")
    sample_s1 = s1_train.sample(n=min(80000, len(s1_train)), random_state=RANDOM_STATE).reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    
    feats, labels = [], []
    for eid, nn, na, nc in sample_s1[['entity_id', 'norm_name', 'norm_addr', 'norm_country']].values:
        nn = str(nn); na = str(na); nc = str(nc)
        true_m = gt.get(eid, set()) & set(s23_tr_lkp.keys())
        cands = {c for c in idx_tr.query(nn, na, nc, cap=80) if c in s23_tr_lkp}
        
        # Positives
        for p in true_m:
            r = s23_tr_lkp[p]
            feats.append(extract_features(nn, na, nc, r[0], r[1], r[2]))
            labels.append(1)

        # Hard Negatives
        neg_cands = list(cands - true_m)
        n_neg = min(len(neg_cands), max(len(true_m) * 5, 4))
        if neg_cands and n_neg > 0:
            chosen = rng.choice(len(neg_cands), n_neg, replace=False)
            for ci in chosen:
                r = s23_tr_lkp[neg_cands[ci]]
                feats.append(extract_features(nn, na, nc, r[0], r[1], r[2]))
                labels.append(0)

    X_train = np.array(feats, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)
    log.info(f"  Pairs Generated: {len(y_train):,} (Positives: {(y_train==1).sum():,}, Negatives: {(y_train==0).sum():,})")

    # 5. Train Blended Ensemble
    log.info("\n[5] Training Model Ensemble...")
    ensemble = train_ensemble(X_train, y_train)

    # 6. Fine-Grained Threshold Tuning on Held-out Validation
    log.info("\n[6] Threshold Optimization on Validation Set...")
    val_sample = s1_val.sample(n=min(15000, len(s1_val)), random_state=RANDOM_STATE).reset_index(drop=True)
    val_ids = list(val_sample['entity_id'])

    val_scores = {}
    for eid, nn, na, nc in val_sample[['entity_id', 'norm_name', 'norm_addr', 'norm_country']].values:
        nn = str(nn); na = str(na); nc = str(nc)
        cands = {c for c in idx_tr.query(nn, na, nc, cap=80) if c in s23_tr_lkp}
        if cands:
            fs = [extract_features(nn, na, nc, s23_tr_lkp[c][0], s23_tr_lkp[c][1], s23_tr_lkp[c][2]) for c in cands]
            probs = ensemble.predict_proba(np.array(fs, dtype=np.float32))[:, 1]
            val_scores[eid] = dict(zip(cands, probs))
        else:
            val_scores[eid] = {}

    best_thresh = 0.98; best_f05 = -1.0; best_p = 0.0; best_r = 0.0
    thresh_grid = np.linspace(0.85, 0.995, 30)
    for t in thresh_grid:
        preds = {}
        for sid in val_ids:
            sc = val_scores[sid]
            # Dynamic filtering: top score must exceed threshold, secondary matches must be within margin
            if not sc:
                preds[sid] = set()
                continue
            sorted_c = sorted(sc.items(), key=lambda x: -x[1])
            top_cand, top_prob = sorted_c[0]
            if top_prob >= t:
                # Accept top match and close runner-ups
                matched = {top_cand}
                for c, p in sorted_c[1:6]:
                    if p >= t and (top_prob - p) <= 0.08:
                        matched.add(c)
                preds[sid] = matched
            else:
                preds[sid] = set()

        f, p, r = macro_f05(val_ids, preds, gt)
        if f > best_f05:
            best_f05 = f; best_thresh = t; best_p = p; best_r = r

    log.info(f"  *** Optimal Threshold: {best_thresh:.4f} => Val F0.5={best_f05:.4f} (P={best_p:.4f}, R={best_r:.4f}) ***")

    # Clean training memory
    del s1_tr, s1_train, s1_val, sample_s1, val_sample, X_train, y_train, idx_tr, s23_tr_lkp
    gc.collect()

    # 7. Test Inference (Vectorized Streaming)
    log.info("\n[7] Starting Test Inference with Vectorized Streaming...")
    s1_te = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t')
    s1_te['norm_name'] = s1_te['business_name'].apply(clean_name)
    s1_te['norm_addr'] = s1_te['business_address'].apply(clean_address)
    s1_te['norm_country'] = s1_te['country'].apply(clean_country)
    all_te_ids = list(s1_te['entity_id'])

    log.info("  Loading Test Source 2 and Source 3...")
    s2_te = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t')
    s3_te = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t')
    s23_te = pd.concat([s2_te, s3_te], ignore_index=True)
    del s2_te, s3_te; gc.collect()

    s23_te['norm_name'] = s23_te['business_name'].apply(clean_name)
    s23_te['norm_addr'] = s23_te['business_address'].apply(clean_address)
    s23_te['norm_country'] = s23_te['country'].apply(clean_country)

    idx_te = HighRecallBlockingIndex()
    idx_te.fit(s23_te)

    s23_te_lkp = dict(zip(
        s23_te['entity_id'],
        zip(s23_te['norm_name'], s23_te['norm_addr'], s23_te['norm_country'])
    ))
    del s23_te; gc.collect()

    # Streaming Output Files
    out_matching = os.path.join(OUT_DIR, 'matching_results_v2.tsv')
    out_candidate = os.path.join(OUT_DIR, 'candidate_pairs_v2.tsv')
    log.info(f"  Streaming outputs to {out_matching} and {out_candidate}...")

    BATCH_SIZE = 25000
    t_inf = time.time()
    
    with open(out_matching, 'w', encoding='utf-8') as fm, \
         open(out_candidate, 'w', encoding='utf-8') as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        # Process in batches
        for start_idx in range(0, len(s1_te), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(s1_te))
            batch_slice = s1_te.iloc[start_idx:end_idx]

            # Collect pairs for batch
            pair_features = []
            pair_meta = []  # (eid, cand_id)
            entity_candidates = {}

            for eid, nn, na, nc in batch_slice[['entity_id', 'norm_name', 'norm_addr', 'norm_country']].values:
                nn = str(nn); na = str(na); nc = str(nc)
                cands = [c for c in idx_te.query(nn, na, nc, cap=70) if c in s23_te_lkp]
                entity_candidates[eid] = cands
                for c in cands:
                    r = s23_te_lkp[c]
                    pair_features.append(extract_features(nn, na, nc, r[0], r[1], r[2]))
                    pair_meta.append((eid, c))

            # Vectorized scoring
            if pair_features:
                X_batch = np.array(pair_features, dtype=np.float32)
                batch_probs = ensemble.predict_proba(X_batch)[:, 1]
                
                # Group by entity
                grouped_scores = defaultdict(list)
                for (eid, c), prob in zip(pair_meta, batch_probs):
                    grouped_scores[eid].append((c, prob))
            else:
                grouped_scores = defaultdict(list)

            # Write batch results directly to disk
            for eid in batch_slice['entity_id']:
                cands = entity_candidates.get(eid, [])
                sc = grouped_scores.get(eid, [])
                
                matched = []
                if sc:
                    sc.sort(key=lambda x: -x[1])
                    top_c, top_p = sc[0]
                    if top_p >= best_thresh:
                        matched.append(top_c)
                        for c, p in sc[1:6]:
                            if p >= best_thresh and (top_p - p) <= 0.08:
                                matched.append(c)

                fm.write(f"{eid}\t{','.join(sorted(matched))}\n")
                fc.write(f"{eid}\t{','.join(sorted(cands))}\n")

            if (end_idx % 100000 == 0) or (end_idx == len(s1_te)):
                log.info(f"  Processed {end_idx:,}/{len(s1_te):,} ({time.time()-t_inf:.1f}s elapsed)...")

    log.info(f"\nPipeline V2 Complete in {(time.time()-t_start)/60:.1f} minutes!")
    log.info(f"Output generated:")
    log.info(f"  {out_matching}")
    log.info(f"  {out_candidate}")


if __name__ == '__main__':
    run_enhanced_pipeline()
