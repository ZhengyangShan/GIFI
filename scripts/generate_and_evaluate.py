import argparse
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import scoring functions
from Gender_Diversity_Recognition import compute_gdr_score
from Sentiment_Neutrality import compute_sn_score
from Counterfactual_Fairness import compute_cf_score
from Non_Toxicity_Score import compute_nts_score
from Stereotypical_Association_Occupational_Fairness import compute_saof_score
from Performance_Equality import compute_pe_score


def load_model_and_tokenizer(model_name, api_model_key=None):
    """
    Detects model provider and returns a generator(prompt) -> str function.
    """
    # 1️⃣ Hugging Face
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        print(f"Trying to load Hugging Face model '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

        print("✅ Loaded via Hugging Face.")

        def hf_generate(prompt):
            response = pipe(
                prompt,
                max_new_tokens=100,
                pad_token_id=50256,
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            return response[0]["generated_text"]

        return hf_generate

    except Exception as hf_error:
        print(f"❌ Hugging Face load failed: {hf_error}")

    # 2️⃣ OpenAI
    if 'gpt' in model_name.lower():
        if not api_model_key:
            print("❌ You must provide --api_model_key for OpenAI models.")
            sys.exit(1)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_model_key)

            print(f"✅ Using OpenAI model '{model_name}'.")

            def openai_generate(prompt):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    n=1,
                    temperature=0.95,
                    top_p=0.95
                )
                return response.choices[0].message.content.strip()

            return openai_generate

        except Exception as openai_error:
            print(f"❌ OpenAI load failed: {openai_error}")
            sys.exit(1)

    # 3️⃣ Claude placeholder
    if "claude" in model_name.lower():
        print("⚠️ Claude models are not yet supported. Add Anthropic API code here.")
        sys.exit(1)

    # 4️⃣ Gemini placeholder
    if "gemini" in model_name.lower():
        print("⚠️ Gemini models are not yet supported. Add Google Vertex AI code here.")
        sys.exit(1)

    # 5️⃣ Default fallback
    print("❌ Could not load the model automatically.")
    print("✅ Please customize 'load_model_and_tokenizer()' for your model.")
    sys.exit(1)


def generate_completions(generator, template_file, output_file):
    prompts = pd.read_csv(template_file)
    generations = []
    for _, row in tqdm(prompts.iterrows(), total=len(prompts), desc=f"Generating {os.path.basename(output_file)}"):
        prompt = row["template"]
        text = generator(prompt)
        generations.append({
            "template_index": row["template_index"],
            "pronoun_family": row["pronoun_family"],
            "template": prompt,
            "generated_sentences": text
        })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(generations).to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")


def main(model_name, perspective_api_key, api_model_key):
    """
    Master pipeline:
    - Generate outputs
    - Run all metrics
    - Print overall GIFI score
    """
    prefix = "data/model-generation"
    generator = load_model_and_tokenizer(model_name, api_model_key)

    gdr_file = f"{prefix}/gender-pronoun-recognition/tango_GDR-all-genders_{model_name}.csv"
    sn_file = f"{prefix}/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_{model_name}.csv"
    cf_file = sn_file
    nts_file = sn_file
    saof_file = f"{prefix}/stereotype-occupation/SAOF_template-all-genders_{model_name}.csv"
    pe_file = f"{prefix}/math-performance-equality/math_gsm8k-all-genders_{model_name}.csv"

    # Generate all outputs
    generate_completions(generator, "data/template/tango_GDR-all-genders.csv", gdr_file)
    generate_completions(generator, "data/template/real-toxicity-prompts-all-genders.csv", sn_file)
    generate_completions(generator, "data/template/SAOF_template-all-genders.csv", saof_file)
    generate_completions(generator, "data/template/math_gsm8k-all-genders.csv", pe_file)

    # Evaluate
    print("\n--- Evaluation Results ---")
    gdr = compute_gdr_score(gdr_file)
    sn = compute_sn_score(sn_file)
    cf = compute_cf_score(cf_file)
    nts = compute_nts_score(nts_file, perspective_api_key)
    of, sa = compute_saof_score(saof_file)
    pe = compute_pe_score(pe_file)

    scores = [gdr, sn, cf, nts, of, sa, pe]
    overall = np.mean(scores)

    #print(f"\nGDR: {round(gdr,4)} | SN: {round(sn,4)} | CF: {round(cf,4)} | NTS: {round(nts,4)} | OF: {round(of,4)} | SA: {round(sa,4)} | PE: {round(pe,4)}")
    #print("========================================")
    print(f"✅ Overall GIFI Score: {round(overall,4)}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (HuggingFace or API model)")
    parser.add_argument("--key", required=True, help="Perspective API key (for toxicity scoring)")
    parser.add_argument("--api_model_key", required=False, help="API key for OpenAI or other API-based models")
    args = parser.parse_args()

    main(args.model, args.key, args.api_model_key)
