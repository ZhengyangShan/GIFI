"""Generate completions for the four GIFI prompt sets with one model, then score them.

Examples (run from the repository root):
  python scripts/generate_and_evaluate.py --model gpt2 --backend hf
  python scripts/generate_and_evaluate.py --model meta-llama/Meta-Llama-3-8B-Instruct --backend hf --chat
  OPENAI_API_KEY=... python scripts/generate_and_evaluate.py --model gpt-4o --backend openai --key PERSPECTIVE_KEY
  python scripts/generate_and_evaluate.py --model my-model --backend openai --base_url http://localhost:8000/v1

Decoding follows the paper: up to 200 new tokens, temperature 0.95, top-p 0.95, one sample per prompt
(the SA/OF prompt file already repeats each prompt 11 times).
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, HERE)

from prompts import build_prompt  # noqa: E402

TASKS = {  # task -> (template file, output subfolder, output prefix, group column)
    "gdr": ("tango-all-genders.csv", "gender-pronoun-recognition", "tango_GDR-all-genders", "template_index"),
    "rtp": ("real-toxicity-prompts-all-genders.csv", "sentiment-toxicity-counterfactual",
            "real-toxicity-prompts-all-genders", "template_index"),
    "saof": ("SAOF_template-all-genders.csv", "stereotype-occupation", "SAOF_template-all-genders", "index"),
    "pe": ("math_gsm8k-all-genders.csv", "math-performance-equality", "math_gsm8k-all-genders", "template_index"),
}
MAX_NEW_TOKENS, TEMPERATURE, TOP_P = 200, 0.95, 0.95


def hf_generator(model_name, chat, batch_size, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()

    def generate(prompts):
        out = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="generate"):
            batch = prompts[i:i + batch_size]
            if chat:
                batch = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True) for p in batch]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=not chat).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                                     top_p=TOP_P, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]  # completion only, never the prompt
            out += [t.strip() for t in tok.batch_decode(new, skip_special_tokens=True)]
        return out

    return generate


def openai_generator(model_name, base_url):
    from openai import OpenAI

    client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def generate(prompts):
        out = []
        for p in tqdm(prompts, desc="generate"):
            r = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": p}],
                                               max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
            out.append((r.choices[0].message.content or "").strip())
        return out

    return generate


def run_task(task, generate, name, chat, template_dir, out_dir, limit=None, overwrite=False):
    tfile, sub, prefix, group = TASKS[task]
    out_path = os.path.join(out_dir, sub, f"{prefix}_{name}.csv")
    if os.path.exists(out_path) and not overwrite:
        print(f"{task}: {out_path} exists, skipping (use --overwrite)")
        return out_path
    df = pd.read_csv(os.path.join(template_dir, tfile))
    if limit:
        df = df[df[group].isin(df[group].drop_duplicates().head(limit))]
    df = df.copy()
    df["prompt"] = [build_prompt(task, t, chat) for t in df["template"]]
    df["generated_sentences"] = generate(df["prompt"].tolist())
    df["model"] = name
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{task}: saved {len(df)} rows to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate GIFI completions and score them")
    ap.add_argument("--model", required=True, help="HF model id or API model name")
    ap.add_argument("--backend", choices=["hf", "openai"], required=True)
    ap.add_argument("--chat", action="store_true", help="HF only: wrap prompts in the model's chat template")
    ap.add_argument("--base_url", help="openai backend: OpenAI-compatible endpoint (vLLM, etc.)")
    ap.add_argument("--name", help="name used in output files (default: last part of --model)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--template_dir", default=os.path.join(ROOT, "data", "template"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "model-generation"))
    ap.add_argument("--limit", type=int, help="only the first N prompt groups per task (smoke test)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--key", help="Perspective API key for NTS")
    ap.add_argument("--skip_eval", action="store_true")
    a = ap.parse_args()

    name = a.name or a.model.rstrip("/").split("/")[-1]
    chat = a.chat or a.backend == "openai"
    generate = (hf_generator(a.model, a.chat, a.batch_size, a.seed) if a.backend == "hf"
                else openai_generator(a.model, a.base_url))
    for t in a.tasks:
        run_task(t, generate, name, chat, a.template_dir, a.out_dir, a.limit, a.overwrite)

    if not a.skip_eval:
        if set(a.tasks) != set(TASKS):
            print("Scoring skipped: run_all needs all four tasks.")
            return
        from run_all import METRICS, run_all

        scores, _ = run_all(name, a.out_dir, a.key)
        print(f"\n=== {name} ===")
        for m in METRICS:
            if m in scores:
                print(f"{m:4s} {scores[m]:.4f}")
        print(f"GIFI {scores['GIFI']:.1f}" if "GIFI" in scores else "GIFI not reported: needs all seven metrics.")


if __name__ == "__main__":
    main()
