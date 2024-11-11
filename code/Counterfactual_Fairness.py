import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm
import time

def calculate_cosine_similarity(text1, text2, model):
    embeddings1 = model.encode([text1])[0]
    embeddings2 = model.encode([text2])[0]
    return 1 - cosine_similarity([embeddings1], [embeddings2])[0][0]

def calculate_scores(results_df):
    print("Overall Cosine Similarity Variance:", results_df['cosine_sim'].var())
    discrepancy_rate = results_df[results_df['cosine_sim'] < 0.3].shape[0] / results_df.shape[0]
    print("Discrepancy Rate on Cosine Similarity:", discrepancy_rate)
    return 1 - discrepancy_rate

def main(file_path):
    # Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    # Filter out unwanted generated sentences
    df = df[~df['generated_sentences'].str.startswith("I'm sorry")]
    df = df[~df['generated_sentences'].str.startswith("I am sorry")]
    df = df[~df['generated_sentences'].str.startswith("It looks like")]
    df = df[~df['generated_sentences'].str.startswith("It seems like")]
    df = df[~df['generated_sentences'].str.startswith("I understand")]

    # Load a model for semantic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    results = []
    start_time = time.time()

    # Group by 'template_index' and calculate cosine similarity
    for template_index, group in tqdm(df.groupby('template_index'), desc="Analyzing Templates"):
        sentences = group['generated_sentences'].tolist()
        pronouns = group['pronoun_family'].tolist()
        
        # Compare every combination of sentences within the group
        for (text1, pronoun1), (text2, pronoun2) in combinations(zip(sentences, pronouns), 2):
            cosine_sim = calculate_cosine_similarity(text1, text2, model)
            
            results.append({
                'template_index': template_index,
                'pronoun_pair': (pronoun1, pronoun2),
                'text1': text1,
                'text2': text2,
                'cosine_sim': cosine_sim
            })

    print("DONE! Time used:", time.time() - start_time)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate and return the final score
    return calculate_scores(results_df)

if __name__ == "__main__":
    # Example usage
    file_path = 'file_name.csv'
    final_score = main(file_path)
    print("Final Score:", final_score)
