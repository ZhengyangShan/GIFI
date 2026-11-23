"""Build the HuggingFace release of the GIFI prompts from data/template/*.csv.

Writes one parquet file per config (gdr, rtp, saof, pe) plus the dataset card:
  <out_dir>/README.md
  <out_dir>/<config>/test.parquet

Column names used by the scoring code (template_index, pronoun_family, template, simple_answer,
index, state) are kept, so a generation frame built from these rows plus a `generated_sentences`
column can be passed straight to the metric functions in code/.

Usage (from the repository root): python scripts/build_hf_dataset.py <out_dir>
"""
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "code"))

from gifi_common import FAMILIES  # noqa: E402
from prompts import build_prompt  # noqa: E402

TEMPLATES = os.path.join(ROOT, "data", "template")
CARD = os.path.join(HERE, "hf_dataset_card.md")
LICENSE = os.path.join(HERE, "hf_LICENSE.md")


def read(name):
    return pd.read_csv(os.path.join(TEMPLATES, name))


def gdr():
    d = read("tango-all-genders.csv")
    out = pd.DataFrame(dict(
        template_index=d.template_index, pronoun_family=d.pronoun_family, template=d.template,
        prompt=[build_prompt("gdr", t) for t in d.template],
        antecedent=d.ante, antecedent_type=d.ante_type, tango_template_idx=d.tango_template_idx.astype(int),
        source_text=d.source_text))
    return out, 200, 11


def rtp():
    d = read("real-toxicity-prompts-all-genders.csv")
    note = d.note.fillna("")
    out = pd.DataFrame(dict(
        template_index=d.template_index, pronoun_family=d.pronoun_family, template=d.template,
        prompt=[build_prompt("rtp", t) for t in d.template],
        source_family=d.source_family, rtp_filename=d.rtp_filename, source_text=d.source_text,
        edited_from_source=note.str.contains("edited_in_v1_cleaning")))
    return out, 200, 11


def saof():
    d = read("SAOF_template-all-genders.csv")
    assert d.groupby("index").template.nunique().eq(1).all()
    d = d.drop_duplicates("index").reset_index(drop=True)
    out = pd.DataFrame(dict(
        index=d["index"], template=d.template, prompt=[build_prompt("saof", t) for t in d.template],
        state=d.state, word=d.word, stereotype=d.stereotype, num_samples=11))
    return out, 200, 1


def pe():
    d = read("math_gsm8k-all-genders.csv")
    out = pd.DataFrame(dict(
        template_index=d.template_index, pronoun_family=d.pronoun_family, template=d.template,
        prompt=[build_prompt("pe", t) for t in d.template], simple_answer=d.simple_answer.astype(int),
        replaced_name=d.replaced_name, source_family=d.source_family, gsm8k_split=d.gsm8k_split,
        source_text=d.source_text))
    return out, 100, 11


def check(name, df, n_items, per_item):
    key = "index" if name == "saof" else "template_index"
    assert df[key].nunique() == n_items, name
    assert len(df) == n_items * per_item, name
    assert df.notna().all().all(), (name, df.columns[df.isna().any()].tolist())
    if "pronoun_family" in df:
        assert df.groupby(key).pronoun_family.apply(lambda s: sorted(s) == sorted(FAMILIES)).all(), name
        assert not df.duplicated([key, "pronoun_family"]).any(), name


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out_dir = sys.argv[1]
    for name, fn in [("gdr", gdr), ("rtp", rtp), ("saof", saof), ("pe", pe)]:
        df, n_items, per_item = fn()
        check(name, df, n_items, per_item)
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        df.to_parquet(os.path.join(out_dir, name, "test.parquet"), index=False)
        print(f"{name}: {len(df)} rows, columns {list(df.columns)}")
    shutil.copy(CARD, os.path.join(out_dir, "README.md"))
    shutil.copy(LICENSE, os.path.join(out_dir, "LICENSE.md"))
    print(f"card -> {os.path.join(out_dir, 'README.md')}")


if __name__ == "__main__":
    main()
