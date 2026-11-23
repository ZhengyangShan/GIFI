"""Score one model's generations on all seven GIFI metrics and print the GIFI index.

Expects the four generation files written by scripts/generate_and_evaluate.py (or the same layout):
  <gen_dir>/gender-pronoun-recognition/tango_GDR-all-genders_<model>.csv
  <gen_dir>/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_<model>.csv
  <gen_dir>/stereotype-occupation/SAOF_template-all-genders_<model>.csv
  <gen_dir>/math-performance-equality/math_gsm8k-all-genders_<model>.csv

NTS needs a Perspective API key unless the RTP file already has a TOXICITY column.
Without either, NTS is skipped and GIFI is not reported.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from gifi_common import load_generations  # noqa: E402
from Gender_Diversity_Recognition import gdr_report  # noqa: E402
from Sentiment_Neutrality import sn_report  # noqa: E402
from Non_Toxicity_Score import nts_report  # noqa: E402
from Counterfactual_Fairness import cf_report  # noqa: E402
from Stereotypical_Association_Occupational_Fairness import saof_report  # noqa: E402
from Performance_Equality import pe_report  # noqa: E402

METRICS = ["GDR", "SN", "NTS", "CF", "SA", "OF", "PE"]


def paths(gen_dir, model):
    return {
        "gdr": f"{gen_dir}/gender-pronoun-recognition/tango_GDR-all-genders_{model}.csv",
        "rtp": f"{gen_dir}/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_{model}.csv",
        "saof": f"{gen_dir}/stereotype-occupation/SAOF_template-all-genders_{model}.csv",
        "pe": f"{gen_dir}/math-performance-equality/math_gsm8k-all-genders_{model}.csv",
    }


def run_all(model, gen_dir="data/model-generation", api_key=None, skip=()):
    p = paths(gen_dir, model)
    for k, f in p.items():
        if not os.path.exists(f):
            raise FileNotFoundError(f"{k}: {f} not found")
    scores, details = {}, {}

    if "GDR" not in skip:
        details["GDR"] = gdr_report(load_generations(p["gdr"]))
        scores["GDR"] = details["GDR"]["score"]
    rtp = load_generations(p["rtp"])
    if "SN" not in skip:
        details["SN"] = sn_report(rtp)[0]
        scores["SN"] = details["SN"]["score"]
    if "NTS" not in skip:
        if api_key or "TOXICITY" in rtp.columns:
            details["NTS"] = nts_report(rtp, api_key)[0]
            scores["NTS"] = details["NTS"]["score"]
        else:
            print("NTS skipped: no Perspective API key and no TOXICITY column.")
    if "CF" not in skip:
        details["CF"] = cf_report(rtp)
        scores["CF"] = details["CF"]["score"]
    if not {"SA", "OF"} <= set(skip):
        r = saof_report(load_generations(p["saof"], required=("index", "template", "state")))
        details["SA"], details["OF"] = r["SA"], r["OF"]
        scores["SA"], scores["OF"] = r["SA"]["score"], r["OF"]["score"]
    if "PE" not in skip:
        details["PE"] = pe_report(load_generations(p["pe"], required=("pronoun_family", "simple_answer")))
        scores["PE"] = details["PE"]["score"]

    if all(m in scores for m in METRICS):
        scores["GIFI"] = 100 * sum(scores[m] for m in METRICS) / len(METRICS)
    return scores, details


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run all GIFI metrics for one model")
    ap.add_argument("--model", required=True, help="model name used in the generation file names")
    ap.add_argument("--gen_dir", default="data/model-generation", help="folder with the four generation subfolders")
    ap.add_argument("--key", help="Perspective API key (for NTS)")
    ap.add_argument("--skip", nargs="*", default=[], choices=METRICS, help="metrics to skip")
    ap.add_argument("--json", help="optional path to save scores and diagnostics as JSON")
    a = ap.parse_args()
    scores, details = run_all(a.model, a.gen_dir, a.key, a.skip)
    print(f"\n=== {a.model} ===")
    for m in METRICS:
        if m in scores:
            print(f"{m:4s} {scores[m]:.4f}")
    if "GIFI" in scores:
        print(f"GIFI {scores['GIFI']:.1f}")
    else:
        print("GIFI not reported: needs all seven metrics.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "scores": scores, "details": details}, f, indent=2, default=str)
