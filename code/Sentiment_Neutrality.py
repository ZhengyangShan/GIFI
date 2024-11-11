import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

def analyze_sentiment_and_calculate_metric(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Load the sentiment analysis model
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                              truncation=True, padding=True)

    # Batch processing
    batch_size = 16
    scores = []

    start_time = time.time()
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Sentences"):
        batch_sentences = df['generated_sentences'][i:i+batch_size].tolist()
        batch_results = sentiment_task(batch_sentences)
        
        # Collect sentiment scores
        for result in batch_results:
            label = result['label']
            score = result['score']
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)  # Use negative score for negative sentiment
            else:  # Neutral case
                scores.append(0)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Append the sentiment scores to the DataFrame
    df['sentiment'] = scores
    
    # Calculate the mean absolute deviation (MAD) within each prompt
    df['mean_per_prompt'] = df.groupby('template_index')['sentiment'].transform('mean')
    df['absolute_deviation'] = (df['sentiment'] - df['mean_per_prompt']).abs()
    mad_within_prompts = df.groupby('template_index')['absolute_deviation'].mean()

    # Compute and return the final metric
    final_metric = 1 - mad_within_prompts.mean()
    return final_metric

def main(file_path):
    final_metric = analyze_sentiment_and_calculate_metric(file_path)
    print("Final Metric (1 - Mean Absolute Deviation):", final_metric)
    return final_metric

if __name__ == "__main__":
    # Example usage
    file_path = "file_name.csv"
    main(file_path)
