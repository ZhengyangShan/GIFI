import argparse
from Gender_Diversity_Recognition import compute_gdr_score
from Sentiment_Neutrality import compute_sn_score
from Counterfactual_Fairness import compute_cf_score
from Non_Toxicity_Score import compute_nts_score
from Stereotypical_Association-Occupational_Fairness import compute_saof_score
from Performance_Equality import compute_pe_score

def run_all(model_name: str, api_key: str):
    prefix = "data/model-generation"
    print("Evaluating model:", model_name)

    gdr_file = f"{prefix}/gender-pronoun-recognition/tango_GDR-all-genders_{model_name}.csv"
    sn_file = f"{prefix}/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_{model_name}.csv"
    cf_file = sn_file
    nts_file = sn_file
    saof_file = f"{prefix}/stereotype-occupation/SAOF_template-all-genders_{model_name}.csv"
    pe_file = f"{prefix}/math-performance-equality/math_gsm8k-all-genders_{model_name}.csv"

    print("GDR Score:", round(compute_gdr_score(gdr_file), 4))
    print("SN Score:", round(compute_sn_score(sn_file), 4))
    print("CF Score:", round(compute_cf_score(cf_file), 4))
    print("NTS Score:", round(compute_nts_score(nts_file, api_key), 4))
    of, sa = compute_saof_score(saof_file)
    print("OF Score:", round(of, 4))
    print("SA Score:", round(sa, 4))
    print("PE Score:", round(compute_pe_score(pe_file), 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--key", required=True, help="Perspective API key")
    args = parser.parse_args()

    run_all(args.model, args.key)
