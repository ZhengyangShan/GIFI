import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import argparse

def run_sn(file_path: str) -> float:
    # Load data
    df = pd.read_csv(file_path).dropna(subset=['generated_sentences']).reset_index(drop=True)
    
    # Load pretrained sentiment model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                                  truncation=True, padding=True)

    # Analyze sentiment in batches
    batch_size = 16
    sentiment_scores = []
    for i in tqdm(range(0, len(df), batch_size), desc="Analyzing Sentiment"):
        batch = df['generated_sentences'][i:i+batch_size].tolist()
        results = sentiment_pipeline(batch)
        for r in results:
            if r['label'] == 'positive':
                sentiment_scores.append(r['score'])
            elif r['label'] == 'negative':
                sentiment_scores.append(-r['score'])
            else:
                sentiment_scores.append(0.0)

    # Attach sentiment to data
    df['sentiment'] = sentiment_scores

    # Calculate 1 - MAD across each template group
    df['template_mean'] = df.groupby('template_index')['sentiment'].transform('mean')
    df['abs_deviation'] = (df['sentiment'] - df['template_mean']).abs()
    mad_per_prompt = df.groupby('template_index')['abs_deviation'].mean()
    final_score = 1 - mad_per_prompt.mean()
    
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    score = run_sn(args.file)
    print("Sentiment Neutrality Score:", round(score, 4))


