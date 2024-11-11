# LLM Gender Inclusivity Fairness Index

This repository contains code for analyzing and measuring gender inclusivity and bias in large language models (LLMs) by introducing the Gender Inclusivity Fairness Index (GIFI). The project leverages datasets and evaluates various metrics based on pronoun distribution. 

## Project Structure

- **Dataset Directory**: `data/`
  - `data/template`: Contains CSV files used for generating model outputs.
  - `data/model-generation/`: Contains CSV files used for analysis.
- **Code Directory**: `code/`
  - `Gender_Diversity_Recognition.py`: Script for gender diversity analysis.
  - `Non-Toxicity-Score.py`: Script for toxicity socres.
  - `Sentiment_Neutrality.py`: Script for sentiment analysis using a pre-trained sentiment model.
  - `Counterfactual_Fairness.py`: Script for semantic similarity analysis.
  - `Stereotypical_Association-Occupational_Fairness.py`: Script to compute OF and SA scores.
  - `Performance_Equality.py`: Script for math performance anlysis.
- **Analysis Directory**: `analysis/`
  - Contains output and analysis results.

## Datasets

- **Folder 1**: `data/model-generation/gender-pronoun-recognition/`
  - Description: Contains generated sentences for gender diversity recognition.
- **Folder 2**: `data/model-generation/sentiment-toxicity-counterfactual/`
  - Description: Contains generated sentences for sentiment, non toxicity and semantic similarity analysis.
- **Folder 3**: `data/model-generation/stereotype-occupation/`
  - Description: Contains generated sentences for stereotypical association (SA) and occupational fairness (OF) analysis.
- **Folder 4**: `data/model-generation/math-performance-equality/`
  - Description: Contains generated sentences for math analysis. 

## Requirements

To set up the environment and install all dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Run

### Pronoun Recognition 
- **Gender Diversity Recognition (GDR)**

```bash
python Gender_Diversity_Recognition.py
```

### Fairness in Distribution 
- **Sentiment Neutrality (SN)**

```bash
python Sentiment_Neutrality.py
```

- **Counterfactual Fairness (CF)**

```bash
python Counterfactual_Fairness.py
```

- **Non-Toxicity Score (NTS)**

```bash
python Non-Toxicity-Score.py
```
### Stereotype and Role Assignment 

- **Stereotype and Occupation (SA & OF)**

```bash
python Stereotypical_Association-Occupational_Fairness.py
```

### Consistency in Performance

- **Performance Equality (PE)**

```bash
python Performance_Equality.py
```


























