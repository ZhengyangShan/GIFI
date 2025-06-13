# Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework for Evaluating Gender Diversity in  Large Language Models

This repository provides code, datasets, and evaluation pipelines for analyzing and measuring gender inclusivity and bias in large language models (LLMs). We introduce the **Gender Inclusivity Fairness Index (GIFI)** — a comprehensive and scalable benchmark for LLMs across binary and non-binary gender representations.

🔍 [Website](https://zhengyangshan.github.io/GIFI.github.io/)

📄 Paper

## 🗣️ Overview 
GIFI evaluates the capacity of LLMs to fairly and accurately handle diverse gender identities. It measures fairness across **seven dimensions**:

| Dimension                        | Abbr | Description                                                                 |
|----------------------------------|------|-----------------------------------------------------------------------------|
| Gender Diversity Recognition     | GDR  | Recognizes and generates correct grammatical usage of diverse pronouns     |
| Sentiment Neutrality            | SN   | Keeps sentiment consistent across gender variations                        |
| Non-Toxicity Score              | NTS  | Avoids toxic language toward binary and non-binary identities             |
| Counterfactual Fairness         | CF   | Produces semantically similar outputs across gender replacements           |
| Stereotypical Association       | SA   | Avoids gender stereotypes in role assignments                              |
| Occupational Fairness           | OF   | Avoids unfair gender bias in occupation mentions                           |
| Performance Equality            | PE   | Maintains consistency in reasoning/maths performance across genders        |

We evaluate **22 models** (open and proprietary) and identify major disparities in gender inclusivity across model families and scales.
<p align="center">
  <img src="figures/GIFI-ranking.png" alt="Image 1" width="45%">
  <img src="figures/GIFI.png" alt="Image 2" width="45%">
</p>

<p align="center">
  <img src="figures/GIFI-individual.png" alt="Image 1" width="100%">
</p>

## ⚙️ Project Structure

```bash
GIFI/
├── data/
│   ├── template/
│   └── model-generation/
├── code/
│   ├── Gender_Diversity_Recognition.py
│   ├── Sentiment_Neutrality.py
│   ├── Non-Toxicity-Score.py
│   ├── Counterfactual_Fairness.py
│   ├── Stereotypical_Association-Occupational_Fairness.py
│   └── Performance_Equality.py
├── figures/
├── requirements.txt
└── README.md
```

- **Dataset Directory**: `data/`
  - `data/template`: template used for generating model outputs.
  - `data/model-generation/`: model generations used for analysis.
- **Code Directory**: `code/`
  - `Gender_Diversity_Recognition.py`: Script for gender diversity analysis.
  - `Non-Toxicity-Score.py`: Script for toxicity socres.
  - `Sentiment_Neutrality.py`: Script for sentiment analysis using a pre-trained sentiment model.
  - `Counterfactual_Fairness.py`: Script for semantic similarity analysis.
  - `Stereotypical_Association-Occupational_Fairness.py`: Script to compute OF and SA scores.
  - `Performance_Equality.py`: Script for math performance anlysis.
- **Figures Directory**: `figures/`
  - Contains output and analysis results.
 

### 🧪 Datasets

All generations are saved under `data/model-generation/`

- **gender-pronoun-recognition**: for gender diversity recognition (GDR).
- **sentiment-toxicity-counterfactual**: for sentiment (SN), non toxicity (NTS) and semantic similarity (CF) analysis.
- **stereotype-occupation**: for stereotypical association (SA) and occupational fairness (OF) analysis.
- **math-performance-equality**: for math analysis (PE). 

## 🧩 Installation

To set up the environment and install all dependencies, run:

```bash
git clone https://github.com/ZhengyangShan/GIFI.git
cd GIFI
pip install -r requirements.txt
```

You may also create a virtual environment:

```bash
python3 -m venv gifi-env
source gifi-env/bin/activate
pip install -r requirements.txt
```

## 🚀 Evaluation Pipeline 

Each script runs independently for evaluating a specific fairness metric

### 🔵 Pronoun Recognition 
- Gender Diversity Recognition (GDR)

Replace MODEL_NAME with acutal model name. 

```bash
python Gender_Diversity_Recognition.py \
  --file data/model-generation/gender-pronoun-recognition/tango_GDR-all-genders_MODEL_NAME.csv
```

### 🟡 Fairness in Distribution 
- Sentiment Neutrality (SN)

```bash
python Sentiment_Neutrality.py
```

- Counterfactual Fairness (CF)

```bash
python Counterfactual_Fairness.py
```

- Non-Toxicity Score (NTS)

Set up your API key at [here](https://perspectiveapi.com).

```bash
python Non-Toxicity-Score.py
```
### 🟣 Stereotype and Role Assignment 

- Stereotype and Occupation (SA & OF)

Replace MODEL_NAME with acutal model name. 

```bash
python Stereotypical_Association-Occupational_Fairness.py \
  --file data/model-generation/stereotype-occupation/SAOF_template-all-genders_MODEL_NAME.csv
```

### 🟠 Consistency in Performance

- Performance Equality (PE)

```bash
python Performance_Equality.py
```


## 🥳 Citations

If you find this useful, please consider citing our work and starring the repo.

















