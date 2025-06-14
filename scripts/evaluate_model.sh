#!/bin/bash

MODEL_NAME=$1
API_KEY=$2

echo "Running evaluation for: $MODEL_NAME"

python Gender_Diversity_Recognition.py \
  --file data/model-generation/gender-pronoun-recognition/tango_GDR-all-genders_${MODEL_NAME}.csv

python Sentiment_Neutrality.py \
  --file data/model-generation/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_${MODEL_NAME}.csv

python Counterfactual_Fairness.py \
  --file data/model-generation/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_${MODEL_NAME}.csv

python Non-Toxicity-Score.py \
  --file data/model-generation/sentiment-toxicity-counterfactual/real-toxicity-prompts-all-genders_${MODEL_NAME}.csv \
  --key $API_KEY

python Stereotypical_Association-Occupational_Fairness.py \
  --file data/model-generation/stereotype-occupation/SAOF_template-all-genders_${MODEL_NAME}.csv

python Performance_Equality.py \
  --file data/model-generation/math-performance-equality/math_gsm8k-all-genders_${MODEL_NAME}.csv
