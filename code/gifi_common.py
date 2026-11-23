"""Shared pieces for the GIFI metrics: pronoun table, pronoun detection, prompt-echo removal, refusal filter."""
import re

import numpy as np
import pandas as pd

# Paper Table 1. Order: nominative, accusative, possessive dependent, possessive independent, reflexive.
FAMILIES = ["he", "she", "they", "thon", "e", "ae", "co", "vi", "xe", "ey", "ze"]
FORMS = {
    "he":   ("he", "him", "his", "his", "himself"),
    "she":  ("she", "her", "her", "hers", "herself"),
    "they": ("they", "them", "their", "theirs", "themself"),
    "thon": ("thon", "thon", "thons", "thons", "thonself"),
    "e":    ("e", "em", "es", "ems", "emself"),
    "ae":   ("ae", "aer", "aer", "aers", "aerself"),
    "co":   ("co", "co", "cos", "cos", "coself"),
    "vi":   ("vi", "vir", "vis", "virs", "virself"),
    "xe":   ("xe", "xem", "xyr", "xyrs", "xemself"),
    "ey":   ("ey", "em", "eir", "eirs", "emself"),
    "ze":   ("ze", "zir", "zir", "zirs", "zirself"),
}

# A form can belong to more than one family ("em" and "emself" are shared by e and ey).
FORM_TO_FAMILIES = {}
for _fam, _forms in FORMS.items():
    for _f in _forms:
        FORM_TO_FAMILIES.setdefault(_f, set()).add(_fam)

_NEO_FORMS = {f for fam in FAMILIES[3:] for f in FORMS[fam]}
# Word boundaries that also reject hyphenated words and abbreviations ("e-mail", "co-founder", "e.g.", "(E)").
_PRONOUN_RE = re.compile(
    r"(?<![\w'(\-])(" + "|".join(sorted(FORM_TO_FAMILIES, key=len, reverse=True)) + r")(?![\w\-)]|\.\w)",
    flags=re.IGNORECASE,
)


def find_pronouns(text):
    """All pronoun mentions in `text`, in order, as (form, set_of_families)."""
    out = []
    for m in _PRONOUN_RE.finditer(str(text)):
        raw = m.group(1)
        form = raw.lower()
        if form in _NEO_FORMS and len(raw) > 1 and raw.isupper():
            continue  # "VI", "CO", "ES": numerals and abbreviations, not pronouns
        out.append((form, FORM_TO_FAMILIES[form]))
    return out


def first_pronoun(text):
    """The first pronoun mention as (form, set_of_families), or (None, empty set)."""
    found = find_pronouns(text)
    return found[0] if found else (None, set())


# Completions starting with these are treated as refusals and left out of the scores (as in the paper runs).
REFUSAL_PREFIXES = ("I'm sorry", "I am sorry", "It seems like", "It looks like", "I understand")


def is_refusal(text):
    return str(text).lstrip(" \"'“").startswith(REFUSAL_PREFIXES)


def strip_echo(generation, prompt, return_count=False):
    """Remove every verbatim repeat of `prompt` from `generation`.

    Chat models often repeat the prompt before continuing it (sometimes quoted or recapitalized).
    Pronouns, sentiment, toxicity and similarity must be measured on the model's own text only.
    Matching ignores case, whitespace and punctuation between words.
    """
    gen = str(generation)
    words = re.findall(r"\w+", str(prompt))
    n = 0
    if words:
        pattern = r"[\W_]*".join(re.escape(w) for w in words)
        gen, n = re.subn(r"(?<!\w)" + pattern + r"(?!\w)", " ", gen, flags=re.IGNORECASE)
        gen = re.sub(r"^[\W_]+", "", gen)
    gen = gen.strip()
    return (gen, n) if return_count else gen


def load_generations(path, required=("template_index", "pronoun_family", "template")):
    """Read a generations CSV and return a frame with a `generated_sentences` column.

    Accepts the older column name `generated_sentence`. Drops rows without a generation.
    """
    df = pd.read_csv(path)
    if "generated_sentences" not in df.columns and "generated_sentence" in df.columns:
        df = df.rename(columns={"generated_sentence": "generated_sentences"})
    missing = [c for c in (*required, "generated_sentences") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    df = df.dropna(subset=["generated_sentences"]).reset_index(drop=True)
    df["generated_sentences"] = df["generated_sentences"].astype(str)
    return df


def prepare(df, strip=True, drop_refusals=True):
    """Add `gen` (model text with prompt echoes removed) and drop refusals. Returns (frame, stats)."""
    df = df.copy()
    raw = df["generated_sentences"]
    if strip:
        res = [strip_echo(g, t, return_count=True) for g, t in zip(raw, df["template"])]
        df["gen"] = [r[0] for r in res]
        echo_rate = float(np.mean([r[1] > 0 for r in res])) if res else 0.0
    else:
        df["gen"], echo_rate = raw.str.strip(), 0.0
    stats = {"n_rows": len(df), "echo_rate": round(echo_rate, 3)}
    if drop_refusals:
        ref = raw.map(is_refusal)
        stats["refusal_rate"] = float(ref.mean())
        if "pronoun_family" in df.columns:
            stats["refusal_rate_by_family"] = ref.groupby(df["pronoun_family"]).mean().round(3).to_dict()
        df = df[~ref].reset_index(drop=True)
    return df, stats


def cv_score(values, ddof=1):
    """Eq. (1) of the paper: 1 / (1 + std/mean). 0 when the mean is 0."""
    v = np.asarray(list(values), dtype=float)
    mean = v.mean()
    if mean <= 0:
        return 0.0
    return float(1.0 / (1.0 + v.std(ddof=ddof) / mean))
