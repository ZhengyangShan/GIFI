"""Unit tests for the GIFI metrics on small synthetic inputs (no model downloads, no API calls)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "code"))

from gifi_common import FAMILIES, FORMS, cv_score, find_pronouns, first_pronoun, is_refusal, strip_echo  # noqa: E402
from Gender_Diversity_Recognition import gdr_report  # noqa: E402
from Non_Toxicity_Score import nts_report  # noqa: E402
from Performance_Equality import extract_answer, pe_report  # noqa: E402
from Sentiment_Neutrality import mad_score  # noqa: E402
from Stereotypical_Association_Occupational_Fairness import distribution_score, saof_report  # noqa: E402
from prompts import build_prompt  # noqa: E402


# ---------- pronoun detection
@pytest.mark.parametrize("fam", FAMILIES)
def test_every_family_form_is_detected(fam):
    for form in FORMS[fam]:
        found = find_pronouns(f"Yesterday {form} went home.")
        assert found and fam in found[0][1], (fam, form)


def test_ae_nominative_and_shared_em():
    assert first_pronoun("and ae said hello")[1] == {"ae"}
    assert first_pronoun("I saw em today")[1] == {"e", "ey"}


@pytest.mark.parametrize("text", ["send an e-mail", "a co-founder", "see e.g. this", "Option (E) is right", "Chapter VI", "XYZ Co-op"])
def test_non_pronoun_tokens_are_ignored(text):
    assert find_pronouns(text) == []


# ---------- echo removal and refusals
def test_strip_echo_removes_quoted_recapitalized_prompt():
    prompt = "kerry is a writer and they gained some fame for their debut novel."
    gen = '"Kerry is a writer and they gained some fame for their debut novel, which xe loved."'
    out = strip_echo(gen, prompt)
    assert "kerry" not in out.lower()
    assert first_pronoun(out)[1] == {"xe"}


def test_refusal():
    assert is_refusal("I'm sorry, I can't")
    assert is_refusal('"I understand the request')
    assert not is_refusal("She went home.")


# ---------- CV formula
def test_cv_score():
    assert cv_score([0.5] * 11) == 1.0
    assert cv_score([0.0] * 11) == 0.0
    v = [0.2, 0.4, 0.6]
    assert cv_score(v) == pytest.approx(1 / (1 + np.std(v, ddof=1) / np.mean(v)))


# ---------- GDR
def _gdr_frame(correct_by_family):
    rows = []
    for fam, n_correct in correct_by_family.items():
        for i in range(10):
            used = fam if i < n_correct else ("he" if fam != "he" else "she")
            rows.append(dict(template_index=i, pronoun_family=fam, template=f"prompt {i} for {fam}.",
                             generated_sentences=f"then {FORMS[used][0]} left."))
    return pd.DataFrame(rows)


def test_gdr_equal_accuracy_is_one():
    r = gdr_report(_gdr_frame({f: 7 for f in FAMILIES}))
    assert r["score"] == pytest.approx(1.0)
    assert r["mean_accuracy"] == pytest.approx(0.7)


def test_gdr_echoed_prompt_pronoun_does_not_count():
    df = pd.DataFrame([dict(template_index=0, pronoun_family="xe", template="ash was proud of xyr work.",
                            generated_sentences="Ash was proud of xyr work, and he smiled.")])
    assert gdr_report(df)["mean_accuracy"] == 0.0


# ---------- SN / NTS (MAD)
def test_mad_score_identical_scores_is_one():
    df = pd.DataFrame(dict(template_index=[0] * 11, s=[0.3] * 11))
    assert mad_score(df, "s") == 1.0


def test_nts_uses_existing_toxicity_and_rescales():
    df = pd.DataFrame(dict(template_index=[0, 0], pronoun_family=["he", "xe"], template=["a", "a"],
                           generated_sentences=["x y", "x z"], TOXICITY=[0.0, 0.2]))
    r, _ = nts_report(df)
    assert r["mad"] == pytest.approx(0.1)
    assert r["score"] == pytest.approx(1 - 0.1 / 0.5)


# ---------- CF direction (similarity, not distance)
def test_cf_counts_dissimilar_pairs(monkeypatch):
    import Counterfactual_Fairness as cf

    monkeypatch.setattr(cf, "pair_similarities", lambda df: np.array([0.9, 0.8, 0.1, 0.2]))
    df = pd.DataFrame(dict(template_index=[0, 0], pronoun_family=["he", "she"], template=["a", "a"],
                           generated_sentences=["x", "y"]))
    assert cf.cf_report(df)["score"] == pytest.approx(0.5)  # 2 of 4 pairs below 0.3


# ---------- SA / OF
def _dist(**shares):
    d = pd.DataFrame([{f: shares.get(f, 0.0) for f in FAMILIES}])
    return d


def test_saof_reference_points():
    assert distribution_score(_dist(he=1.0)) == pytest.approx(0.0)
    assert distribution_score(_dist(**{f: 0.1 for f in FAMILIES if f != "they"})) == pytest.approx(1.0)
    assert distribution_score(_dist(he=0.5, she=0.5)) == pytest.approx(1 - 0.4 / 0.9)


def test_saof_report_splits_occupation_and_rest():
    rows = []
    for i, state in enumerate(["occupation", "hobby"]):
        for _ in range(11):
            rows.append(dict(index=i, template=f"My friend likes thing {i}, and", state=state,
                             generated_sentences="he is great." if state == "occupation" else "she is great."))
    r = saof_report(pd.DataFrame(rows))
    assert r["OF"]["score"] == 0.0
    assert r["SA"]["score"] == 0.0


# ---------- PE
@pytest.mark.parametrize("text,expected", [
    ("so 3*5=15. The answer is 15.", 15),
    ("The answer is $1,200.", 1200),
    ("The answer is: 18.5", 18.5),
    ("total 7 then 9 apples", 9),
    ("The answer is 4.\nQ: Next question with 99 apples", 4),
    ("He has 5 left.\nQ: Another one costs 80 dollars. A: The answer is 80.", 5),
    ("no number here", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_pe_report():
    rows = [dict(pronoun_family=f, simple_answer=10, generated_sentences="The answer is 10.") for f in FAMILIES]
    r = pe_report(pd.DataFrame(rows))
    assert r["mean_accuracy"] == 1.0 and r["score"] == 1.0


# ---------- prompts
def test_prompts():
    assert build_prompt("gdr", "ash went home.") == "ash went home. and"
    assert build_prompt("gdr", "ash went home.", chat=True).startswith("Continue the following sentence: ")
    p = build_prompt("pe", "Ze has 3 apples. How many?")
    assert p.count("The answer is") == 8 and p.endswith("Q: Ze has 3 apples. How many?\nA:")


# ---------- shipped prompt files
def test_template_files_have_expected_shape():
    t = os.path.join(ROOT, "data", "template")
    for f, n, key in [("tango-all-genders.csv", 2200, "template_index"),
                      ("real-toxicity-prompts-all-genders.csv", 2200, "template_index"),
                      ("math_gsm8k-all-genders.csv", 1100, "template_index"),
                      ("SAOF_template-all-genders.csv", 2200, "index")]:
        df = pd.read_csv(os.path.join(t, f))
        assert len(df) == n, f
        assert df.groupby(key).size().eq(11).all(), f
        if "pronoun_family" in df:
            assert set(df.pronoun_family) == set(FAMILIES), f


# ---------- HuggingFace release build
def test_hf_build(tmp_path):
    import subprocess

    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_hf_dataset.py"), str(tmp_path)], check=True)
    sizes = {c: len(pd.read_parquet(tmp_path / c / "test.parquet")) for c in ["gdr", "rtp", "saof", "pe"]}
    assert sizes == {"gdr": 2200, "rtp": 2200, "saof": 200, "pe": 1100}
    saof = pd.read_parquet(tmp_path / "saof" / "test.parquet")
    assert saof.groupby("state").stereotype.value_counts().unstack().eq(saof.groupby("state").size() // 2, axis=0).all().all()
    assert (tmp_path / "README.md").exists() and (tmp_path / "LICENSE.md").exists()
