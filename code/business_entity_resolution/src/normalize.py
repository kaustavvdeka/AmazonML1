"""
normalize.py - Text normalization utilities for Business Entity Resolution.

Handles business name normalization, address normalization, country normalization,
token extraction, and character n-gram generation.
"""

import re
import unicodedata
from typing import List, Set


# Legal suffix normalization map
LEGAL_SUFFIXES = [
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
    (r'\bgroup\b', 'grp'),
    (r'\binternational\b', 'intl'),
    (r'\bnational\b', 'natl'),
    (r'\bassociates\b', 'assoc'),
    (r'\bassociate\b', 'assoc'),
    (r'\bconsultants\b', 'consult'),
    (r'\bconsultant\b', 'consult'),
    (r'\btraders\b', 'trd'),
    (r'\btrader\b', 'trd'),
    (r'\bindustries\b', 'ind'),
    (r'\bindustry\b', 'ind'),
    (r'\bsolutions\b', 'sol'),
    (r'\bsolution\b', 'sol'),
    (r'\btechnologies\b', 'tech'),
    (r'\btechnology\b', 'tech'),
    (r'\bmanagement\b', 'mgmt'),
    (r'\bdevelopment\b', 'dev'),
    (r'\bfoundation\b', 'fdn'),
    (r'\bhospital\b', 'hosp'),
    (r'\bschool\b', 'sch'),
    (r'\bcentres\b', 'ctr'),
    (r'\bcentre\b', 'ctr'),
    (r'\bcenters\b', 'ctr'),
    (r'\bcenter\b', 'ctr'),
    (r'\bcompany\b', 'co'),
    (r'\bdba\b', ''),
]

# Address abbreviation map
ADDR_ABBREVS = [
    (r'\bstreet\b', 'st'),
    (r'\broad\b', 'rd'),
    (r'\bavenue\b', 'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\bdrive\b', 'dr'),
    (r'\bcourt\b', 'ct'),
    (r'\blane\b', 'ln'),
    (r'\bplace\b', 'pl'),
    (r'\bsuite\b', 'ste'),
    (r'\bapartment\b', 'apt'),
    (r'\bbuilding\b', 'bldg'),
    (r'\bfloor\b', 'fl'),
    (r'\bnortheast\b', 'ne'),
    (r'\bnorthwest\b', 'nw'),
    (r'\bsoutheast\b', 'se'),
    (r'\bsouthwest\b', 'sw'),
    (r'\bnorth\b', 'n'),
    (r'\bsouth\b', 's'),
    (r'\beast\b', 'e'),
    (r'\bwest\b', 'w'),
    (r'\bpost\s*office\s*box\b', 'po box'),
    (r'\bp\.?o\.?\s*box\b', 'po box'),
    (r'\bnear\b', ''),
    (r'\bopposite\b', ''),
    (r'\bbehind\b', ''),
    (r'\badjacent\s*to\b', ''),
    (r'\bnext\s*to\b', ''),
]


def normalize_unicode(text: str) -> str:
    """NFKD Unicode normalization - preserves all scripts."""
    try:
        return unicodedata.normalize('NFKD', text)
    except Exception:
        return text


def normalize_name(name: str) -> str:
    """
    Normalize a business name:
    1. NFKD Unicode decomposition
    2. Lowercase
    3. & -> and
    4. Legal suffix standardization
    5. Punctuation -> space
    6. Whitespace collapse
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    text = normalize_unicode(name).lower()
    text = text.replace('&', ' and ')
    for pattern, repl in LEGAL_SUFFIXES:
        text = re.sub(pattern, repl, text)
    # Keep word chars + spaces + hyphens (works for Latin + Devanagari + others)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_address(addr: str) -> str:
    """
    Normalize a business address:
    1. NFKD Unicode decomposition
    2. Lowercase
    3. Address abbreviation standardization
    4. Punctuation -> space
    5. Whitespace collapse
    """
    if not isinstance(addr, str) or not addr.strip():
        return ""
    text = normalize_unicode(addr).lower()
    for pattern, repl in ADDR_ABBREVS:
        text = re.sub(pattern, ' ' + repl + ' ', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_country(country: str) -> str:
    """Normalize country string."""
    if not isinstance(country, str):
        return ""
    return country.lower().strip()


def get_name_tokens(norm_name: str) -> Set[str]:
    """Token set from normalized name (min length 2)."""
    return {t for t in norm_name.split() if len(t) >= 2}


def get_addr_tokens(norm_addr: str) -> Set[str]:
    """Token set from normalized address (min length 3)."""
    return {t for t in norm_addr.split() if len(t) >= 3}


def get_bigrams(text: str) -> Set[str]:
    """Character bigrams from text (spaces stripped)."""
    s = text.replace(' ', '')
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()


def get_trigrams(text: str) -> Set[str]:
    """Character trigrams from text (spaces stripped)."""
    s = text.replace(' ', '')
    return {s[i:i+3] for i in range(len(s)-2)} if len(s) >= 3 else set()


def sorted_tokens(norm_name: str) -> str:
    """Token-sorted form for order-invariant comparison."""
    return ' '.join(sorted(norm_name.split()))


def extract_numbers(text: str) -> List[str]:
    """Extract all numeric tokens from text."""
    return re.findall(r'\b\d+\b', text)
