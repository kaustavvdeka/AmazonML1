#!/usr/bin/env python3
"""
pipeline.py  –  Amazon ML Challenge 2026: Business Entity Resolution
=====================================================================
Memory-efficient, high-performance pipeline:
  - Vectorised pandas normalization
  - Two-pass blocking index: build frequency map → filter high-freq keys
  - Max-postings filter on query to avoid candidate explosion
  - Phase A (train) → Phase B (test) to control peak RAM

Usage (from student_resource/):
    python3 code/business_entity_resolution/src/pipeline.py
"""

import gc, json, logging, os, re, sys, time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

RANDOM_STATE = 42
SRC       = os.path.dirname(os.path.abspath(__file__))
BASE      = os.path.dirname(os.path.dirname(os.path.dirname(SRC)))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE, 'dataset', 'test')
OUT_DIR   = os.path.join(BASE, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# Strategy F (country+number) removed: generates 50K-800K candidates at 10M scale
# Reordered E→A→B→D: fill budget with small discriminative keys first
FINAL_STRATEGIES = ['E', 'A', 'B', 'D']

# ─────────────────────────────────────────────────────────────
# NORMALIZATION (vectorised)
# ─────────────────────────────────────────────────────────────

_LEGAL = [
    (re.compile(r'\blimited liability company\b'), 'llc'),
    (re.compile(r'\blimited liability partnership\b'), 'llp'),
    (re.compile(r'\bpvt\s*ltd\b'), 'pvtltd'),
    (re.compile(r'\bprivate\s*limited\b'), 'pvtltd'),
    (re.compile(r'\bpvt\s*limited\b'), 'pvtltd'),
    (re.compile(r'\bprivate\s*ltd\b'), 'pvtltd'),
    (re.compile(r'\bincorporated\b'), 'inc'),
    (re.compile(r'\bcorporation\b'), 'corp'),
    (re.compile(r'\blimited\b'), 'ltd'),
    (re.compile(r'\bprivate\b'), 'pvt'),
    (re.compile(r'\benterprises?\b'), 'ent'),
    (re.compile(r'\bservices?\b'), 'svc'),
    (re.compile(r'\bgroup\b'), 'grp'),
    (re.compile(r'\binternational\b'), 'intl'),
    (re.compile(r'\bassociates?\b'), 'assoc'),
    (re.compile(r'\bconsultants?\b'), 'consult'),
    (re.compile(r'\btraders?\b'), 'trd'),
    (re.compile(r'\bindustr(?:y|ies)\b'), 'ind'),
    (re.compile(r'\bsolutions?\b'), 'sol'),
    (re.compile(r'\btechnolog(?:y|ies)\b'), 'tech'),
    (re.compile(r'\bmanagement\b'), 'mgmt'),
    (re.compile(r'\bcentres?\b|\bcenters?\b'), 'ctr'),
    (re.compile(r'\bcompany\b'), 'co'),
    (re.compile(r'\bdba\b'), ''),
]
_ADDR = [
    (re.compile(r'\bstreet\b'), 'st'),
    (re.compile(r'\broad\b'), 'rd'),
    (re.compile(r'\bavenue\b'), 'ave'),
    (re.compile(r'\bboulevard\b'), 'blvd'),
    (re.compile(r'\bdrive\b'), 'dr'),
    (re.compile(r'\bcourt\b'), 'ct'),
    (re.compile(r'\blane\b'), 'ln'),
    (re.compile(r'\bplace\b'), 'pl'),
    (re.compile(r'\bsuite\b'), 'ste'),
    (re.compile(r'\bapartment\b'), 'apt'),
    (re.compile(r'\bbuilding\b'), 'bldg'),
    (re.compile(r'\bnortheast\b'), 'ne'),
    (re.compile(r'\bnorthwest\b'), 'nw'),
    (re.compile(r'\bsoutheast\b'), 'se'),
    (re.compile(r'\bsouthwest\b'), 'sw'),
    (re.compile(r'\bnorth\b'), 'n'),
    (re.compile(r'\bsouth\b'), 's'),
    (re.compile(r'\beast\b'), 'e'),
    (re.compile(r'\bwest\b'), 'w'),
    (re.compile(r'\bpost\s*office\s*box\b|\bp\.?o\.?\s*box\b'), 'po box'),
    (re.compile(r'\b(?:near|opposite|behind|adjacent\s*to|next\s*to)\b'), ''),
]
_PUNC = re.compile(r'[^\w\s\-]')
_WS   = re.compile(r'\s+')
_NUMS = re.compile(r'\b(\d+)\b')


def _vsub(s: pd.Series, pats: list) -> pd.Series:
    for p, r in pats:
        s = s.str.replace(p, r, regex=True)
    return s


def norm_name_series(s: pd.Series) -> pd.Series:
    s = s.fillna('').str.lower()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = _vsub(s, _LEGAL)
    s = s.str.replace(_PUNC, ' ', regex=True)
    return s.str.replace(_WS, ' ', regex=True).str.strip()


def norm_addr_series(s: pd.Series) -> pd.Series:
    s = s.fillna('').str.lower()
    s = _vsub(s, _ADDR)
    s = s.str.replace(_PUNC, ' ', regex=True)
    return s.str.replace(_WS, ' ', regex=True).str.strip()


def load_and_normalize(path: str, nrows=None) -> pd.DataFrame:
    """Load TSV and add norm_name, norm_addr, norm_country columns."""
    df = pd.read_csv(path, sep='\t', dtype=str, nrows=nrows,
                     keep_default_na=False, na_values=[''])
    df.fillna('', inplace=True)
    df['norm_name']    = norm_name_series(df['business_name'])
    df['norm_addr']    = norm_addr_series(df['business_address'])
    df['norm_country'] = df['country'].str.lower().str.strip()
    return df


def load_gt(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep='\t', dtype=str,
                     keep_default_na=False, na_values=[''])
    df.fillna('', inplace=True)
    gt = {}
    for r in df.itertuples(index=False):
        mids = str(r.matched_entity_ids).strip()
        gt[r.source1_entity_id] = \
            [x.strip() for x in mids.split(',') if x.strip()] if mids else []
    return gt


# ─────────────────────────────────────────────────────────────
# BLOCKING INDEX with max-postings filter
# ─────────────────────────────────────────────────────────────

# Per-strategy max-postings guards. Keys with more postings than the
# threshold are too common to be discriminative — skip them.
MAX_POSTINGS_NAME   = 1500   # B: name tokens  (less common on 10M data)
MAX_POSTINGS_ADDR   = 500    # D: addr tokens  (very common — road/nagar/st)
MAX_POSTINGS_PREFIX = 500    # E: country|prefix4
MAX_CANDS_PER_ENTITY = 100   # hard cap; inner-loop break also enforces this


class BlockingIndex:
    """
    Four-strategy blocking index (E→A→B→D order).
    F (country+number) removed: generates 50K-800K candidates at 10M scale.
    Strategies run in ascending candidate-count order so the budget (100) is
    filled with the most discriminative keys first.

    Strategy  Key                  Max-postings guard
    ────────  ───────────────────  ──────────────────
    E         country|name_pfx4   MAX_POSTINGS_PREFIX (500)
    A         exact norm name     none  (unique by nature)
    B         name token ≥2 ch    MAX_POSTINGS_NAME   (1500)
    D         addr token ≥3 ch    MAX_POSTINGS_ADDR   (500)
    """

    def __init__(self):
        self.A: Dict[str, list] = defaultdict(list)
        self.B: Dict[str, list] = defaultdict(list)
        self.C: Dict[str, list] = defaultdict(list)
        self.D: Dict[str, list] = defaultdict(list)
        self.E: Dict[str, list] = defaultdict(list)
        self.F: Dict[str, list] = defaultdict(list)
        self._n: int = 0   # total records indexed

    def build_from_df(self, df: pd.DataFrame) -> None:
        """Build all six indexes from normalised dataframe."""
        log.info(f'  Building blocking index over {len(df):,} records...')
        t0 = time.time()
        self._n = len(df)

        for eid, nm, ad, cn in df[['entity_id','norm_name',
                                    'norm_addr','norm_country']].values:
            nm = str(nm); ad = str(ad); cn = str(cn)

            # A
            if nm:
                self.A[nm].append(eid)
            # B
            for tok in nm.split():
                if len(tok) >= 2:
                    self.B[tok].append(eid)
            # C
            core = nm.replace(' ', '')[:30]
            for j in range(len(core) - 1):
                self.C[core[j:j+2]].append(eid)
            # D
            for tok in ad.split():
                if len(tok) >= 3:
                    self.D[tok].append(eid)
            # E
            pfx = nm[:4].strip()
            if cn and pfx:
                self.E[cn + '|' + pfx].append(eid)
            # F
            if cn:
                for n in _NUMS.findall(ad)[:3]:
                    self.F[cn + '|' + n].append(eid)

        log.info(f'  Index built in {time.time()-t0:.1f}s  '
                 f'(A={len(self.A):,} B={len(self.B):,} '
                 f'C={len(self.C):,} D={len(self.D):,})')

    def query(self, nm: str, ad: str, cn: str,
              strategies: List[str] = FINAL_STRATEGIES) -> Set[str]:
        """
        Union candidates across strategies with max-postings guards.
        Inner-loop breaks ensure the per-entity budget is respected
        WITHIN each strategy, not just between strategies.
        """
        out: Set[str] = set()
        cap = MAX_CANDS_PER_ENTITY

        for s in strategies:
            if s == 'E':
                pfx = nm[:4].strip()
                if cn and pfx:
                    pl = self.E.get(cn + '|' + pfx, [])
                    if len(pl) <= MAX_POSTINGS_PREFIX:
                        out.update(pl)

            elif s == 'A':
                if nm:
                    out.update(self.A.get(nm, []))

            elif s == 'B':
                for tok in nm.split():
                    if len(tok) >= 2:
                        pl = self.B.get(tok, [])
                        if len(pl) <= MAX_POSTINGS_NAME:
                            out.update(pl)
                    if len(out) >= cap:   # inner-loop budget check
                        break

            elif s == 'D':
                for tok in ad.split():
                    if len(tok) >= 3:
                        pl = self.D.get(tok, [])
                        if len(pl) <= MAX_POSTINGS_ADDR:
                            out.update(pl)
                    if len(out) >= cap:   # inner-loop budget check
                        break

            if len(out) >= cap:           # outer strategy-level budget check
                break

        return out


# ─────────────────────────────────────────────────────────────
# FEATURES (30-dim)
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'name_exact','name_lev','name_wratio','name_token_set','name_token_sort',
    'name_jaccard','name_overlap','name_bigram_jac','name_trigram_jac',
    'name_len_diff','name_common_tok','name_ntok_s1','name_ntok_s2',
    'addr_exact','addr_lev','addr_token_set','addr_jaccard','addr_overlap',
    'addr_bigram_jac','addr_num_jac','addr_len_diff',
    'addr_both_empty','addr_one_empty',
    'country_exact','country_both_have',
    'name_x_addr','weighted_combined','max_name_sim',
    'has_common_name_tok','has_common_addr_tok',
]
N_FEAT = 30


def _jac(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _ovlp(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))


def _bg(s: str) -> set:
    c = s.replace(' ', '')
    return {c[i:i+2] for i in range(len(c)-1)} if len(c) >= 2 else set()


def _tg(s: str) -> set:
    c = s.replace(' ', '')
    return {c[i:i+3] for i in range(len(c)-2)} if len(c) >= 3 else set()


def feat(nn1, na1, nc1, nn2, na2, nc2) -> List[float]:
    """30-feature vector for one candidate pair."""
    f = []
    # Name (13)
    f.append(float(nn1 == nn2 and nn1 != ''))
    f.append(fuzz.ratio(nn1, nn2) / 100)
    f.append(fuzz.WRatio(nn1, nn2) / 100)
    f.append(fuzz.token_set_ratio(nn1, nn2) / 100)
    f.append(fuzz.token_sort_ratio(nn1, nn2) / 100)
    t1n = {t for t in nn1.split() if len(t)>=2}
    t2n = {t for t in nn2.split() if len(t)>=2}
    ctok = len(t1n & t2n)
    f.append(_jac(t1n, t2n))
    f.append(_ovlp(t1n, t2n))
    f.append(_jac(_bg(nn1), _bg(nn2)))
    f.append(_jac(_tg(nn1), _tg(nn2)))
    l1n, l2n = len(nn1), len(nn2)
    f.append(abs(l1n-l2n) / max(l1n+l2n,1))
    f.append(float(ctok))
    f.append(float(len(t1n)))
    f.append(float(len(t2n)))
    # Address (10)
    f.append(float(na1 == na2 and na1 != ''))
    f.append(fuzz.ratio(na1, na2) / 100)
    f.append(fuzz.token_set_ratio(na1, na2) / 100)
    t1a = {t for t in na1.split() if len(t)>=3}
    t2a = {t for t in na2.split() if len(t)>=3}
    catok = len(t1a & t2a)
    f.append(_jac(t1a, t2a))
    f.append(_ovlp(t1a, t2a))
    f.append(_jac(_bg(na1[:50]), _bg(na2[:50])))
    f.append(_jac(set(_NUMS.findall(na1)), set(_NUMS.findall(na2))))
    l1a, l2a = len(na1), len(na2)
    f.append(abs(l1a-l2a) / max(l1a+l2a,1))
    f.append(float(l1a==0 and l2a==0))
    f.append(float((l1a==0) != (l2a==0)))
    # Country (2)
    f.append(float(nc1==nc2))
    f.append(float(bool(nc1 and nc2)))
    # Combined (5)
    ns=f[3]; as_=f[15]; cs=f[23]
    f.append(ns*as_)
    f.append(0.6*ns + 0.3*as_ + 0.1*cs)
    f.append(max(f[1],f[2],f[3],f[4]))
    f.append(float(ctok>0))
    f.append(float(catok>0))
    return f


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def f05(p, r):
    d = 0.25*p + r
    return 1.25*p*r/d if d > 0 else 0.0


def macro_f05(ids, preds, gt):
    fs, ps, rs = [], [], []
    for sid in ids:
        ts = set(gt.get(sid, []))
        pr = preds.get(sid, set())
        if not ts and not pr:
            fs.append(1.0); ps.append(1.0); rs.append(1.0)
        elif not ts:
            fs.append(0.0); ps.append(0.0); rs.append(1.0)
        elif not pr:
            fs.append(0.0); ps.append(1.0); rs.append(0.0)
        else:
            tp=len(ts&pr); p_=tp/len(pr); r_=tp/len(ts)
            fs.append(f05(p_,r_)); ps.append(p_); rs.append(r_)
    return float(np.mean(fs)), float(np.mean(ps)), float(np.mean(rs))


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class RuleBasedMatcher:
    def predict_proba(self, X):
        s = np.clip(X[:,26], 0, 1)
        return np.column_stack([1-s, s])


class ScaledLR:
    def __init__(self):
        self.sc = StandardScaler()
        self.m  = LogisticRegression(C=1.0, max_iter=1000,
                                     class_weight='balanced',
                                     random_state=RANDOM_STATE)
    def fit(self, X, y):
        self.m.fit(self.sc.fit_transform(X), y); return self
    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))


def _sp(y): return float((y==0).sum()) / max(float((y==1).sum()), 1.0)


def train_rf(X, y):
    m = RandomForestClassifier(n_estimators=200, max_depth=12,
        min_samples_leaf=3, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X, y); return m


def train_xgb(X, y):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=_sp(y),
        eval_metric='logloss', verbosity=0, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X, y); return m


def train_lgb(X, y):
    m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=_sp(y),
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    m.fit(X, y); return m


TRAINERS = {
    'Rule-based':   lambda X, y: RuleBasedMatcher(),
    'LogReg':       lambda X, y: ScaledLR().fit(X, y),
    'RandomForest': train_rf,
    'XGBoost':      train_xgb,
    'LightGBM':     train_lgb,
}


# ─────────────────────────────────────────────────────────────
# TRAINING PAIR GENERATION
# ─────────────────────────────────────────────────────────────

def generate_pairs(s1_df, idx, gt, s23_lkp,
                   neg_ratio=6, strategies=FINAL_STRATEGIES):
    rng   = np.random.default_rng(RANDOM_STATE)
    feats = []
    labs  = []
    for eid, nn, na, nc in s1_df[['entity_id','norm_name',
                                   'norm_addr','norm_country']].values:
        nn=str(nn); na=str(na); nc=str(nc)
        true_ms = set(gt.get(eid, []))
        cands   = idx.query(nn, na, nc, strategies)

        for m in true_ms:
            r = s23_lkp.get(m)
            if r:
                feats.append(feat(nn,na,nc,r[0],r[1],r[2]))
                labs.append(1)

        neg_pool = [c for c in cands if c not in true_ms and c in s23_lkp]
        n_neg    = min(neg_ratio*max(len(true_ms),1), len(neg_pool))
        if n_neg > 0:
            for ci in rng.choice(len(neg_pool), n_neg, replace=False):
                r = s23_lkp[neg_pool[ci]]
                feats.append(feat(nn,na,nc,r[0],r[1],r[2]))
                labs.append(0)

    X = np.array(feats, dtype=np.float32)
    y = np.array(labs,  dtype=np.int32)
    log.info(f'  Pairs: {len(y):,}  pos={y.sum():,}  neg={(y==0).sum():,}')
    return X, y


# ─────────────────────────────────────────────────────────────
# THRESHOLD SEARCH
# ─────────────────────────────────────────────────────────────

def threshold_search(s1_val, idx, model, gt, s23_lkp, strategies):
    log.info('  Scoring validation candidates...')
    s1_ids = list(s1_val['entity_id'])
    all_sc = {}

    for eid, nn, na, nc in s1_val[['entity_id','norm_name',
                                    'norm_addr','norm_country']].values:
        nn=str(nn); na=str(na); nc=str(nc)
        cands = {c for c in idx.query(nn,na,nc,strategies) if c in s23_lkp}
        if cands:
            fs   = [feat(nn,na,nc,s23_lkp[c][0],s23_lkp[c][1],s23_lkp[c][2])
                    for c in cands]
            prob = model.predict_proba(np.array(fs, dtype=np.float32))[:,1]
            all_sc[eid] = dict(zip(cands, prob))
        else:
            all_sc[eid] = {}

    coarse = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,
              0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    table  = []
    best_f = -1.0; best_t = 0.50

    def _ev(t):
        preds = {sid: {c for c,p in all_sc[sid].items() if p>=t}
                 for sid in s1_ids}
        f,p,r = macro_f05(s1_ids, preds, gt)
        return dict(threshold=round(t,4), f05=round(f,4),
                    precision=round(p,4), recall=round(r,4),
                    n_matches=sum(len(v) for v in preds.values()))

    for t in coarse:
        row=_ev(t); table.append(row)
        if row['f05']>best_f: best_f,best_t=row['f05'],t

    for t in np.linspace(max(0.01,best_t-0.08), min(0.99,best_t+0.08), 17):
        if any(abs(t-r['threshold'])<0.002 for r in table): continue
        row=_ev(t); table.append(row)
        if row['f05']>best_f: best_f,best_t=row['f05'],t

    table.sort(key=lambda x: x['threshold'])
    return best_t, table


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────

def write_outputs(preds, cands, all_ids):
    mp = os.path.join(OUT_DIR, 'matching_results.tsv')
    cp = os.path.join(OUT_DIR, 'candidate_pairs.tsv')
    with open(mp,'w',encoding='utf-8') as fm, \
         open(cp,'w',encoding='utf-8') as fc:
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        fc.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in all_ids:
            ms = preds.get(sid, set())
            cs = cands.get(sid, set()) | ms
            fm.write(f"{sid}\t{','.join(sorted(ms))}\n")
            fc.write(f"{sid}\t{','.join(sorted(cs))}\n")
    n_w = sum(1 for v in preds.values() if v)
    n_t = sum(len(v) for v in preds.values())
    log.info(f'  {n_w:,} S1 with matches  |  {n_t:,} total links')


# ─────────────────────────────────────────────────────────────
# EDA + BLOCKING EXPERIMENTS
# ─────────────────────────────────────────────────────────────

def run_eda(s1, s2, s3, gt):
    log.info('\n' + '='*60 + '\nEDA\n' + '='*60)
    res = {}
    for nm, df in [('S1',s1),('S2',s2),('S3',s3)]:
        n  = len(df)
        mn = (df['business_name']=='').sum()
        ma = (df['business_address']=='').sum()
        cd = df['country'].value_counts().to_dict()
        log.info(f'{nm}: {n:,}  miss_name={mn:,}  miss_addr={ma:,}  countries={cd}')
        res[nm] = dict(n=n, miss_name=int(mn), miss_addr=int(ma),
                       countries={k:int(v) for k,v in cd.items()})
    n_s1=len(gt); sings=sum(1 for v in gt.values() if not v)
    total=sum(len(v) for v in gt.values())
    oto=sum(1 for v in gt.values() if len(v)==1)
    otm=sum(1 for v in gt.values() if len(v)>1)
    s2m=sum(1 for v in gt.values() for m in v if m.startswith('S2-'))
    s3m=sum(1 for v in gt.values() for m in v if m.startswith('S3-'))
    mc=[len(v) for v in gt.values()]
    log.info(f'GT: {n_s1:,} S1 | singletons={sings:,} ({100*sings/n_s1:.1f}%) | '
             f'total={total:,} | oto={oto:,} | otm={otm:,} | '
             f'S2={s2m:,} S3={s3m:,} | max={max(mc)} mean={np.mean(mc):.2f}')
    res['gt']=dict(n_s1=n_s1,singletons=sings,total_matches=total,
                   oto=oto,otm=otm,s2m=s2m,s3m=s3m)
    return res


def blocking_experiments(samp, idx, gt, n_s23):
    configs = {
        'A: exact name':     ['A'],
        'B: name tokens':    ['B'],
        'C: name bigrams':   ['C'],
        'D: addr tokens':    ['D'],
        'E: cntry+pfx4':     ['E'],
        'F: cntry+num':      ['F'],
        'AB':                ['A','B'],
        'ABCDE':             ['A','B','C','D','E'],
        'ABCDEF (union)':    ['A','B','C','D','E','F'],
        'BDF':               ['B','D','F'],
    }
    results = []
    log.info(f"\n{'Strategy':<22} {'Recall':>8} {'AvgC':>8} {'MedC':>8} {'MaxC':>8} {'Reduc':>10}")
    log.info('-'*68)
    for name, strats in configs.items():
        found=0; total_true=0; total_c=0; counts=[]
        for _, row in samp.iterrows():
            ts = set(gt.get(row['entity_id'],[]))
            cs = idx.query(row['norm_name'], row['norm_addr'],
                           row['norm_country'], strats)
            found      += len(ts & cs)
            total_true += len(ts)
            total_c    += len(cs)
            counts.append(len(cs))
        n = len(samp)
        rec = found / max(total_true,1)
        avg = total_c / max(n,1)
        med = float(np.median(counts))
        mx  = int(max(counts)) if counts else 0
        red = 1.0 - total_c / max(n*n_s23,1)
        log.info(f'{name:<22} {rec:>8.4f} {avg:>8.1f} {med:>8.1f} {mx:>8} {red:>10.6f}')
        results.append(dict(strategy=name,recall=rec,avg=avg,med=med,mx=mx,red=red))
    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    t0_all = time.time()
    log.info('='*70)
    log.info('Amazon ML Challenge 2026 – Business Entity Resolution')
    log.info('='*70)

    # ══ PHASE A: TRAINING ════════════════════════════════════

    # [A1] Load train S1 + GT
    log.info('\n[A1] Loading train S1 + GT...')
    t0 = time.time()
    s1_tr = load_and_normalize(os.path.join(TRAIN_DIR,'train_source1.tsv'))
    gt    = load_gt(os.path.join(TRAIN_DIR,'train_ground_truth.tsv'))
    log.info(f'  S1={len(s1_tr):,}  GT={len(gt):,}  ({time.time()-t0:.1f}s)')

    # [A2] Load train S2+S3, build index, free DataFrames
    log.info('\n[A2] Loading train S2+S3 + building index...')
    t0 = time.time()
    s2_tr = load_and_normalize(os.path.join(TRAIN_DIR,'train_source2.tsv'))
    log.info(f'  S2 loaded: {len(s2_tr):,}  ({time.time()-t0:.1f}s)')
    t0 = time.time()
    s3_tr = load_and_normalize(os.path.join(TRAIN_DIR,'train_source3.tsv'))
    log.info(f'  S3 loaded: {len(s3_tr):,}  ({time.time()-t0:.1f}s)')

    run_eda(s1_tr, s2_tr, s3_tr, gt)

    s23_tr = pd.concat([s2_tr, s3_tr], ignore_index=True)
    n_s23_tr = len(s23_tr)
    idx_tr = BlockingIndex()
    idx_tr.build_from_df(s23_tr)

    log.info('  Building S23 lookup...')
    s23_tr_lkp = dict(
        zip(s23_tr['entity_id'],
            zip(s23_tr['norm_name'], s23_tr['norm_addr'], s23_tr['norm_country']))
    )
    del s2_tr, s3_tr, s23_tr; gc.collect()
    log.info(f'  Lookup: {len(s23_tr_lkp):,} entries')

    # [A3] Validation split
    log.info('\n[A3] Validation split (20%)...')
    all_ids   = list(s1_tr['entity_id'])
    has_match = [1 if gt.get(eid) else 0 for eid in all_ids]
    tr_ids, va_ids = train_test_split(all_ids, test_size=0.2,
                                       stratify=has_match, random_state=RANDOM_STATE)
    tr_set, va_set = set(tr_ids), set(va_ids)
    s1_train = s1_tr[s1_tr['entity_id'].isin(tr_set)].reset_index(drop=True)
    s1_val   = s1_tr[s1_tr['entity_id'].isin(va_set)].reset_index(drop=True)
    log.info(f'  Train: {len(s1_train):,}  Val: {len(s1_val):,}')

    # [A4] Blocking experiments (5k sample)
    log.info('\n[A4] Blocking experiments (5k sample)...')
    bl_samp = s1_val.sample(n=min(5000,len(s1_val)), random_state=RANDOM_STATE)
    block_res = blocking_experiments(bl_samp, idx_tr, gt, n_s23_tr)

    # [A5] Training pairs (60k S1 sample)
    log.info('\n[A5] Generating training pairs (60k S1 sample)...')
    s1_samp = s1_train.sample(n=min(60000,len(s1_train)),
                               random_state=RANDOM_STATE).reset_index(drop=True)
    t0 = time.time()
    X_tr, y_tr = generate_pairs(s1_samp, idx_tr, gt, s23_tr_lkp,
                                  neg_ratio=6)
    log.info(f'  Pairs done: {time.time()-t0:.1f}s  X={X_tr.shape}')

    # [A6] Train models + threshold search
    log.info('\n[A6] Train models + threshold search...')
    VAL_N = min(12000, len(s1_val))
    s1_val_eval = s1_val.sample(n=VAL_N, random_state=RANDOM_STATE).reset_index(drop=True)

    mres = {}
    best_name='Rule-based'; best_f05_v=-1.0; best_thresh=0.5

    for mname, trainer in TRAINERS.items():
        log.info(f'\n  === {mname} ===')
        t0=time.time()
        model=trainer(X_tr, y_tr)
        tr_t=time.time()-t0

        t0=time.time()
        best_t, ttable = threshold_search(s1_val_eval, idx_tr, model,
                                           gt, s23_tr_lkp, FINAL_STRATEGIES)
        inf_t=time.time()-t0

        best_row = max(ttable, key=lambda x: x['f05'])
        log.info(f'  thresh={best_t:.4f}  F0.5={best_row["f05"]:.4f}  '
                 f'P={best_row["precision"]:.4f}  R={best_row["recall"]:.4f}  '
                 f'train={tr_t:.1f}s  search={inf_t:.1f}s')

        log.info(f"  {'Thresh':>8} {'F0.5':>8} {'P':>8} {'R':>8} {'#Match':>9}")
        for row in sorted(ttable, key=lambda x: x['threshold']):
            log.info(f"  {row['threshold']:>8.4f} {row['f05']:>8.4f} "
                     f"{row['precision']:>8.4f} {row['recall']:>8.4f} "
                     f"{row['n_matches']:>9,}")

        mres[mname]=dict(model=model, best_threshold=best_t,
                          best_f05=best_row['f05'],
                          precision=best_row['precision'],
                          recall=best_row['recall'],
                          train_time=tr_t, inf_time=inf_t,
                          thresh_table=ttable)
        if best_row['f05'] > best_f05_v:
            best_f05_v=best_row['f05']
            best_name=mname; best_thresh=best_t

    log.info(f'\n  *** BEST: {best_name}  F0.5={best_f05_v:.4f}  thresh={best_thresh:.4f} ***')

    # Feature importance (XGBoost)
    xgb_model = mres.get('XGBoost',{}).get('model')
    if xgb_model and hasattr(xgb_model,'feature_importances_'):
        imp = xgb_model.feature_importances_
        top = sorted(zip(FEATURE_NAMES,imp), key=lambda x:-x[1])[:10]
        log.info('\nXGBoost top-10 feature importance:')
        for fn,fv in top: log.info(f'  {fn:<30} {fv:.4f}')

    # [A7] Retrain best model on full training data
    log.info(f'\n[A7] Retrain {best_name} on full training set (120k sample)...')
    s1_full = s1_tr.sample(n=min(120000,len(s1_tr)),
                            random_state=RANDOM_STATE).reset_index(drop=True)
    t0=time.time()
    X_f, y_f = generate_pairs(s1_full, idx_tr, gt, s23_tr_lkp, neg_ratio=5)
    log.info(f'  Full pairs: {time.time()-t0:.1f}s  X={X_f.shape}')
    t0=time.time()
    final_model = TRAINERS[best_name](X_f, y_f)
    log.info(f'  Final model trained: {time.time()-t0:.1f}s')

    # Free training data
    del s1_tr, s1_train, s1_val, s1_samp, s1_full, s1_val_eval
    del X_tr, y_tr, X_f, y_f, idx_tr, s23_tr_lkp
    gc.collect()

    # ══ PHASE B: TEST INFERENCE ══════════════════════════════

    log.info('\n[B1] Loading test S1...')
    t0=time.time()
    s1_te = load_and_normalize(os.path.join(TEST_DIR,'test_source1.tsv'))
    log.info(f'  S1 test: {len(s1_te):,}  ({time.time()-t0:.1f}s)')

    log.info('\n[B2] Loading test S2+S3 + building index...')
    t0=time.time()
    s2_te = load_and_normalize(os.path.join(TEST_DIR,'test_source2.tsv'))
    log.info(f'  S2 test: {len(s2_te):,}  ({time.time()-t0:.1f}s)')
    t0=time.time()
    s3_te = load_and_normalize(os.path.join(TEST_DIR,'test_source3.tsv'))
    log.info(f'  S3 test: {len(s3_te):,}  ({time.time()-t0:.1f}s)')

    s23_te = pd.concat([s2_te,s3_te], ignore_index=True)
    idx_te = BlockingIndex()
    idx_te.build_from_df(s23_te)
    s23_te_lkp = dict(
        zip(s23_te['entity_id'],
            zip(s23_te['norm_name'], s23_te['norm_addr'], s23_te['norm_country']))
    )
    del s2_te, s3_te, s23_te; gc.collect()
    log.info(f'  Test S23 lookup: {len(s23_te_lkp):,} entries')

    log.info(f'\n[B3] Test inference (threshold={best_thresh:.4f})...')
    t0=time.time()
    te_preds={}; te_cands={}
    all_te_ids=list(s1_te['entity_id']); total=len(s1_te)

    # PRE-FLIGHT: write all-empty placeholder so output is always submission-valid
    # even if inference is interrupted. Will be overwritten at end.
    log.info('  Writing placeholder outputs (all-empty)...')
    write_outputs({}, {}, all_te_ids)
    log.info('  Starting per-entity inference...')

    for i,(eid,nn,na,nc) in enumerate(
            s1_te[['entity_id','norm_name','norm_addr','norm_country']].values):
        if i % 100000 == 0:
            log.info(f'  {i:,}/{total:,}  ({time.time()-t0:.0f}s elapsed)...')
        nn=str(nn); na=str(na); nc=str(nc)
        cands = {c for c in idx_te.query(nn,na,nc,FINAL_STRATEGIES)
                 if c in s23_te_lkp}
        te_cands[eid]=cands
        if not cands:
            te_preds[eid]=set(); continue
        fs   = [feat(nn,na,nc,s23_te_lkp[c][0],
                     s23_te_lkp[c][1],s23_te_lkp[c][2]) for c in cands]
        prob = final_model.predict_proba(np.array(fs,dtype=np.float32))[:,1]
        te_preds[eid] = {c for c,p in zip(cands,prob) if p>=best_thresh}

    log.info(f'  Test inference done: {time.time()-t0:.1f}s')

    log.info('\n[B4] Writing outputs...')
    write_outputs(te_preds, te_cands, all_te_ids)

    # ══ SUMMARY ══════════════════════════════════════════════

    log.info('\n'+'='*70+'\nMODEL COMPARISON\n'+'='*70)
    log.info(f"  {'Model':<14} {'P':>8} {'R':>8} {'F0.5':>8} {'Thresh':>8} {'TrT':>7} {'InfT':>7}")
    log.info('  '+'-'*62)
    for mn,res in mres.items():
        mk=' ◀ BEST' if mn==best_name else ''
        log.info(f"  {mn:<14} {res['precision']:>8.4f} {res['recall']:>8.4f} "
                 f"{res['best_f05']:>8.4f} {res['best_threshold']:>8.4f} "
                 f"{res['train_time']:>6.1f}s {res['inf_time']:>6.1f}s{mk}")

    log.info(f'\nTotal time: {(time.time()-t0_all)/60:.1f} min')
    log.info('\n'+'='*70+'\nBEST VALIDATED PIPELINE\n'+'='*70)
    br=mres[best_name]
    log.info(f'  Pipeline:   Norm → Block(A+B+C+D+E+F) → 30 feats → {best_name}')
    log.info(f'  Threshold:  {best_thresh:.4f}')
    log.info(f'  Val F0.5:   {best_f05_v:.4f}')
    log.info(f'  Val P:      {br["precision"]:.4f}')
    log.info(f'  Val R:      {br["recall"]:.4f}')
    log.info(f'  Output:     {OUT_DIR}/matching_results.tsv')
    log.info(f'              {OUT_DIR}/candidate_pairs.tsv')

    summary=dict(
        best_model=best_name, best_threshold=best_thresh,
        val_f05=best_f05_v,
        val_precision=br['precision'], val_recall=br['recall'],
        model_comparison={k:{kk:vv for kk,vv in v.items()
                             if kk not in ('model','thresh_table')}
                          for k,v in mres.items()},
        blocking_experiments=block_res,
    )
    with open(os.path.join(OUT_DIR,'experiment_summary.json'),'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f'  Summary:    {OUT_DIR}/experiment_summary.json')


if __name__ == '__main__':
    main()
