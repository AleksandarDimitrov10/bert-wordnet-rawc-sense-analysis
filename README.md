# BERT, WordNet, and Meaning Similarity

This project explores whether BERT contextual embeddings capture differences in word meaning across contexts. It uses the RAW-C dataset from Trott & Bergen (2021), where homonymous words appear in sentence pairs with either the same or different meanings.

The project compares contextual word embeddings from BERT with WordNet sense embeddings and human judgments of meaning relatedness.

## Project Overview

Human words are often ambiguous. For example, the word *bank* can refer to a financial institution or to the side of a river. This project investigates whether BERT represents these meaning differences in a way that aligns with human judgments.

The analysis focuses on:

- Extracting contextual embeddings for homonymous words from BERT
- Comparing different BERT layers: static, layer 4, layer 8, and layer 12
- Computing sense embeddings from WordNet glosses
- Measuring cosine similarity between contextual word embeddings and WordNet sense embeddings
- Partially replicating Trott & Bergen's RAW-C experiment by correlating BERT similarity with human meaning-relatedness judgments

## Main Tasks

### 1. Contextual Word Embeddings

BERT embeddings were extracted for target homonyms appearing in four sentence contexts:

- `M1_a`
- `M1_b`
- `M2_a`
- `M2_b`

The project compares embeddings from the static layer and BERT layers 4, 8, and 12.

### 2. WordNet Sense Embeddings

For each target word, WordNet synsets and glosses were extracted. Each gloss was embedded using BERT, producing sense-level representations that could be compared with contextual word embeddings.

### 3. Similarity Analysis

Cosine similarity was used to compare contextual word embeddings with WordNet sense embeddings. This allowed analysis of which WordNet senses were closest to each contextual use of a word.

### 4. RAW-C Partial Replication

The project partially replicated Trott & Bergen's RAW-C analysis by computing cosine similarities between same-meaning and different-meaning sentence pairs, then correlating BERT similarity scores with human meaning-relatedness judgments.

## Key Findings

The results showed that higher BERT layers aligned more strongly with human meaning-relatedness judgments:

- Layer 4: lower alignment
- Layer 8: stronger alignment
- Layer 12: strongest alignment

Layer 12 showed the strongest Spearman correlation with human judgments, suggesting that later BERT layers better capture contextual meaning differences.

However, matching BERT embeddings directly to WordNet senses remained difficult. This suggests that BERT captures graded meaning similarity better than exact symbolic sense categories.

## Technologies Used

- Python
- BERT
- Hugging Face Transformers
- PyTorch
- NLTK / WordNet
- Pandas
- NumPy
- SciPy
- Matplotlib / Seaborn
- RAW-C dataset

## Project Structure

```text
bert-wordnet-meaning-similarity/
├── README.md
├── requirements.txt
├── src/
│   └── cl_group_assignment.py
├── output/
│   └── README.md
└── figures/
    └── README.md
