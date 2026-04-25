# BERT, WordNet, and Meaning Similarity

This project explores whether BERT contextual embeddings capture differences in word meaning across contexts. It uses the RAW-C dataset from Trott & Bergen (2021), where homonymous words appear in sentence pairs with either the same or different meanings.

The project compares contextual word embeddings from BERT with WordNet sense embeddings and human judgments of meaning relatedness.

## Project Overview

Human words are often ambiguous. For example, the word *bank* can refer to a financial institution or to the side of a river. This project investigates whether BERT represents these meaning differences in a way that aligns with human judgments.

The analysis focuses on:

- Extracting contextual embeddings for homonymous words from BERT
- Comparing different BERT layers: static layer, layer 4, layer 8, and layer 12
- Computing sense embeddings from WordNet glosses
- Measuring cosine similarity between contextual word embeddings and WordNet sense embeddings
- Partially replicating Trott & Bergen's RAW-C experiment by correlating BERT similarity with human meaning-relatedness judgments

## Main Tasks

### 1. Contextual Word Embeddings

BERT embeddings were extracted for target homonyms appearing in four sentence contexts:

- M1_a
- M1_b
- M2_a
- M2_b

The project compares embeddings from the static layer and BERT layers 4, 8, and 12.

### 2. WordNet Sense Embeddings

For each target word, WordNet synsets and glosses were extracted. Each gloss was embedded using BERT, producing sense-level representations that could be compared with contextual word embeddings.

### 3. Similarity Analysis

Cosine similarity was used to compare contextual word embeddings with WordNet sense embeddings. This allowed analysis of which WordNet senses were closest to each contextual use of a word.

### 4. RAW-C Partial Replication

The project partially replicated Trott & Bergen's RAW-C analysis by computing cosine similarities between same-meaning and different-meaning sentence pairs, then correlating BERT similarity scores with human meaning-relatedness judgments.

## Key Findings

The results showed that higher BERT layers aligned more strongly with human meaning-relatedness judgments:

- Layer 4 showed weaker alignment
- Layer 8 showed stronger alignment
- Layer 12 showed the strongest alignment

Layer 12 had the strongest Spearman correlation with human judgments, suggesting that later BERT layers better capture contextual meaning differences.

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
- Matplotlib
- Seaborn
- RAW-C dataset

## Project Structure

bert-wordnet-meaning-similarity/
- README.md
- requirements.txt
- .gitignore
- src/
  - bert_wordnet_similarity_analysis.py
- output/
  - README.md
- figures/
  - README.md

## How to Run

This project was originally developed as a notebook-style Computational Linguistics assignment.

### 1. Clone the repository

git clone https://github.com/AleksandarDimitrov10/bert-wordnet-meaning-similarity.git

cd bert-wordnet-meaning-similarity

### 2. Install the required packages

pip install -r requirements.txt

### 3. Run the analysis

Open the Python script in Jupyter, Google Colab, or another notebook-style environment and run the sections in order:

- Task 1: Compute BERT contextual word embeddings
- Task 2: Compute WordNet sense embeddings
- Task 3: Compare contextual embeddings with sense embeddings
- Task 4: Correlate BERT similarities with human relatedness judgments

The script clones the RAW-C dataset and the psycho-embeddings repository when executed.

## My Contribution

This was a Computational Linguistics project focused on connecting linguistic theory with transformer-based NLP methods.

My contribution involved implementing and running the embedding pipeline, computing contextual BERT embeddings across multiple layers, extracting WordNet sense glosses, calculating cosine similarities, and interpreting how well model-derived similarity scores aligned with human judgments of word meaning relatedness.

I also contributed to the written interpretation of the results, especially the comparison between BERT layers, WordNet sense representations, and human semantic judgments.

## Limitations

This project is exploratory and focuses on a partial replication of Trott & Bergen's RAW-C work. It should not be interpreted as a full production-level word sense disambiguation system.

The analysis is strongest as a computational linguistics experiment investigating how transformer representations relate to lexical meaning and human semantic judgments.

## References

Trott, S., & Bergen, B. (2021). RAW-C: Relatedness of Ambiguous Words in Context. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics.

RAW-C dataset: https://github.com/sashakenjeeva/raw-c
