#!/usr/bin/env python
# coding: utf-8

# In human language, words do not always have a fixed meaning. The most striking example is homonymous words: words that have the same form, but very different meanings. For instance, the word "bank", which has a different meaning in the context "I went to the bank to get some money" and "At the river bank, I met my old friend". Polysemous words are words that have different -- yet related -- meanings: for example, "chicken" is the same 'entity' in "My pet chicken is lovely" and "I am having roast chicken for dinner", but has very different meanings in these two contexts. In general, context can modulate almost any word's meaning. This poses a challenge in computational linguistics, as we need to find a way to differentiate among different meanings like humans do. Much research, resources, and models have been put forward to help with this challenge.
# 
# In this assignment, you are going to focus on [Trott and Bergen's (2021)](https://aclanthology.org/2021.acl-long.550/) RAW-C dataset: you are going to conduct a number of explorations with this dataset and partially replicate their research by the end of the assignment. In short, the authors explore how good LLMs are at capturing same/different meanings of words across contexts by comparing it to human judgements. To better understand the idea and the research, start by reading the paper.
# 
# This assignment entails a series of (interconnected) tasks (altogether worth 95 points):
# 
# * **Task 1**. Compute contextual word embeddings at different layers from Trott & Bergen's dataset. Here, each word is found in 4 sentences: 2 with one meaning, 2 with another meaning.
# * **Task 2**. Compute sense embeddings for words in Trott & Bergen's dataset using WordNet, so you have an embedding for each definition of the word.
# * **Task 3**. Compute the similarity between the contextual word embeddings of the homonyms at different layers and their sense embeddings; explore the relationship between homonyms and dominant senses quantitatively and qualitatively
# * **Task 4**. Replicate part of Trott & Bergen's work by computing similarities across sentences with same/different meanings at the different layers and correlate with human similarities; visualise the results and reflect on them
# 
# In order to better understand the assignment, we recommend going through it all before starting so that it is clear how each part is connected to the next (which will help you make decisions about data structures, for instance).

# # Task 1: Compute contextual word embeddings for homonyms [20 points]

# ## Task 1.1: read, explore and extract the necessary data [5 points]

# First, you will have to (fork and) clone the github repository that stores the data you'll need. This can be found here: https://github.com/sashakenjeeva/raw-c . The repo also includes a README with a description of the original files in the repository, as well as some notes relevant for this assignment specifically.

# In[12]:


get_ipython().system('git clone https://github.com/sashakenjeeva/raw-c.git')


# Make sure you mount the drive now so that you have access to the folder (think about setting the working directory in a way that is convenient).

# In[13]:


# --- Google Colab: uncomment the two lines below ---
# from google.colab import drive
# drive.mount('/content/drive')
print("Drive mount skipped (running locally)")


# In[14]:


import os

# Colab: use Drive so files survive session resets
# SAVE_DIR = "/content/drive/MyDrive/CL_Assignment"

# Local: save next to the notebook
SAVE_DIR = os.path.join(os.getcwd(), "output")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}")


# Now, you will have to read the data and organise it in a structure that works for the next parts of the assignment.
# 
# Read and explore the dataframe to see its structure (print part of it). What we need from it are the homonyms (in the form that they appear in the sentence -- the lexeme -- and in their regular form -- the lemma) and their corresponding sentences with different meanings (M1_a and M1_b have same meaning; M2_a, M2_b have same meaning). We only will need the stimuli that are in the final RAW-C dataset, as this is what we'll replicate at the end.
# 
# You can decide which data structure to use, but make sure that all these pieces of information are there (the word, the string, the meaning id, and the corresponding sentences) and easy to retrieve. Show your data at the end, as well as how many stimuli you end up with.

# In[2]:


import pandas as pd

# Load the full stimuli file and the final RAW-C dataset
stimuli_df = pd.read_csv("raw-c/data/stims/stimuli.csv")
rawc_df    = pd.read_csv("raw-c/data/processed/raw-c.csv")

# Keep only the words that appear in the final RAW-C dataset
final_words = set(rawc_df["word"].unique())
stimuli_filtered = stimuli_df[stimuli_df["Word"].isin(final_words)].reset_index(drop=True)

# Explore the dataframe
print("=== Filtered stimuli dataframe (first 3 rows) ===")
print(stimuli_filtered[["String", "Word", "M1_a", "M1_b", "M2_a", "M2_b"]].head(3).to_string())
print(f"\nTotal rows in filtered stimuli: {len(stimuli_filtered)}")

# Build data structure: list of dicts, one entry per (word, meaning_id) pair
# String = lexeme (word form in sentence), Word = lemma (base form)
meaning_ids = ["M1_a", "M1_b", "M2_a", "M2_b"]

stimuli = []
for _, row in stimuli_filtered.iterrows():
    for meaning_id in meaning_ids:
        stimuli.append({
            "lexeme":     row["String"],     # form as it appears in sentence
            "lemma":      row["Word"],        # base/dictionary form
            "meaning_id": meaning_id,
            "sentence":   row[meaning_id]
        })

print(f"\n=== Data structure (first 5 entries) ===")
for entry in stimuli[:5]:
    print(entry)

print(f"\nTotal stimuli entries : {len(stimuli)}")
print(f"Unique words (lemmas) : {len({s['lemma'] for s in stimuli})}")


# ## Task 1.2: Compute the contextualised word embeddings [15 points]
# 

# Now that you have the homonyms and their corresponding sentences, we will need to compute word embeddings for each of them. For this we will use the BERT base model, in its uncased version.
# 
# That is, for each homonym, you will have to compute four embeddings: one for the homonym in M1_a, one in M1_b, one in M2_a, one in M2_b. However, we also want to look into different layers of the BERT model to see which one captures the homonym's meaning best: you want to calculate embeddings at the static layer and at layers 4, 8, 12.
# 
# We will use the package psycho-embeddings (you will use it in class), which allows us to specify which target words we want to obtain the embeddings of, in which sentences, and at which layers, among other things. Make sure to read the documentation of the package so that you know the meaning of the arguments and which ones will come useful to you.
# 
# First of all, install the psycho-embeddings package below.

# In[3]:


# psycho-embeddings is not on PyPI — clone and add to path directly
import sys, os

if not os.path.isdir("psycho-embeddings"):
    get_ipython().system('git clone https://github.com/MilaNLProc/psycho-embeddings.git -q')

# Add to sys.path so the kernel can find it regardless of pip environment
pkg_path = os.path.join(os.getcwd(), "psycho-embeddings")
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

print("psycho-embeddings path added:", pkg_path)


# Now, import the relevant module/function from psycho-embeddings and load the required BERT model.

# In[4]:


from psycho_embeddings import ContextualizedEmbedder

# bert-base-uncased (lower-cases all input, matching the assignment requirement)
embedder = ContextualizedEmbedder("bert-base-uncased", max_length=128)
print("Model loaded successfully.")


# Now, test that everything works correctly by computing an embedding for the word "assignment" in the sentence "I am having so much fun with this assignment!", at static layer and layers 4, 8 and 12 (hint: think of tokenisation and how the embedder deals with that).

# In[5]:


# Test embedding for "assignment"
# averaging=True handles subword tokenisation: if BERT splits a word into multiple
# subword tokens (e.g. "assign" + "##ment"), their vectors are averaged into one.
test_result = embedder.embed(
    words=["assignment"],
    target_texts=["I am having so much fun with this assignment!"],
    layers_id=[4, 8, 12],      # contextual layers
    batch_size=1,
    averaging=True,
    return_static=True,        # adds key -1 for the non-contextual static embedding
    show_progress=False,
)

# The output is a dict: {-1: [static_emb], 4: [layer4_emb], 8: [...], 12: [...]}
# Each value is a list of numpy arrays, one per input word.
print("Output keys (layer ids):", list(test_result.keys()))
for layer_key, label in [(-1, "static"), (4, "layer 4"), (8, "layer 8"), (12, "layer 12")]:
    emb = test_result[layer_key][0]
    print(f"  {label:8s} → shape: {emb.shape},  first 5 values: {emb[:5].round(4)}")


# The next step is to calculate embeddings for the homonyms and their sentences that we got from the RAW-C dataset.
# 
# Make sure that your final output includes the word, the meaning id (M1_a, etc), the corresponding sentence and the embeddings at static layer and layers 4, 8, 12. You should maximally optimise this process by calculating in batches (again, check psycho-embeddings documentation), but keep in mind this might still take a while. First test your pipeline with a small number of inputs, and only run the full scale embedding extraction once you're positive the code works as expected.
# 
# When done, save the output in [pickle](https://docs.python.org/3/library/pickle.html) format (this is similar to json, but it can also handle np.arrays), so that you can easily load it later when needed and do not have to run it again. After pickle dumping (that's the word for saving it in pickle format), print it so that you are sure everything was saved correctly.
# 
# Then, check that your final data includes everything that you need by checking the entry "bank" and print the data pertaining to "bank".

# In[7]:


import pickle, os

# Fallback: define SAVE_DIR if cell 7 was skipped after a kernel restart
if 'SAVE_DIR' not in globals():
    SAVE_DIR = os.path.join(os.getcwd(), "output")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"SAVE_DIR set to: {SAVE_DIR}")

BATCH_SIZE = 32   # process this many (word, sentence) pairs at a time

# --- quick smoke test on 4 entries first ---
test_batch = stimuli[:4]
test_out = embedder.embed(
    words=[s["lexeme"] for s in test_batch],
    target_texts=[s["sentence"] for s in test_batch],
    layers_id=[4, 8, 12],
    batch_size=BATCH_SIZE,
    averaging=True,
    return_static=True,
    show_progress=False,
)
print("Smoke test passed — output keys:", list(test_out.keys()))
print("Embedding shape per entry:", test_out[-1][0].shape)

# --- full extraction in batches ---
cwe_data = []   # final output: one dict per (word, meaning_id) entry

for start in range(0, len(stimuli), BATCH_SIZE):
    batch = stimuli[start : start + BATCH_SIZE]
    result = embedder.embed(
        words=[s["lexeme"] for s in batch],
        target_texts=[s["sentence"] for s in batch],
        layers_id=[4, 8, 12],
        batch_size=BATCH_SIZE,
        averaging=True,
        return_static=True,
        show_progress=False,
    )
    for i, stimulus in enumerate(batch):
        cwe_data.append({
            "lexeme":     stimulus["lexeme"],
            "lemma":      stimulus["lemma"],
            "meaning_id": stimulus["meaning_id"],

            "sentence":   stimulus["sentence"],
            "embeddings": {
                "static": result[-1][i],
                "4":      result[4][i],
                "8":      result[8][i],
                "12":     result[12][i],
            }
        })
    print(f"  Processed {min(start + BATCH_SIZE, len(stimuli))}/{len(stimuli)} entries...", end="\r")

print(f"\nDone. Total entries: {len(cwe_data)}")

# --- pickle dump to Google Drive so it survives session resets ---
CWE_PATH = f"{SAVE_DIR}/cwe_data.pkl"
with open(CWE_PATH, "wb") as f:
    pickle.dump(cwe_data, f)
print(f"Saved to {CWE_PATH}")

# --- verify by reloading and printing "bank" ---
with open(CWE_PATH, "rb") as f:
    cwe_data_loaded = pickle.load(f)

bank_entries = [e for e in cwe_data_loaded if e["lemma"] == "bank"]
print(f"\n=== Entries for 'bank' ({len(bank_entries)} total) ===")
for entry in bank_entries:
    print(f"  meaning_id: {entry['meaning_id']}")
    print(f"  sentence  : {entry['sentence']}")
    print(f"  emb shapes: static={entry['embeddings']['static'].shape}, "
          f"L4={entry['embeddings']['4'].shape}, "
          f"L8={entry['embeddings']['8'].shape}, "
          f"L12={entry['embeddings']['12'].shape}")
    print()


# # Task 2: Compute sense embeddings for the homonym dataset using WordNet [20 points]
# 
# Your next task is to fetch the definitions (glosses) of the homonyms, and compute an embedding for each gloss (each gloss is associated with a specific sense). We do that so we can later see whether the contextualised embeddings computed above represent the meaning of the homonym in context well (by comparing it to the sense embeddings). Figure 18.9 in [Jurafsky's and Martin's (2021) chapter 18](https://web.stanford.edu/~jurafsky/slp3/old_sep21/18.pdf) graphically illustrates this idea. Use this chapter for this part of the assignment, as it will come useful for you both theoretically and practically.
# 
# ## Task 2.1: Fetch senses and glosses for a word [5 points]
# 
# First of all, you will have to figure out how [WordNet](https://www.nltk.org/howto/wordnet.html) works within the nltk package (hint: pay attention to what a synset is).
# 
# Install and import all the necessary components and define a function to extract the glosses of a word and create a dictionary with senses and glosses.
# 
# Then use the word "bat" to test that everything is working correctly: i.e., for "bat", you should be able to get its senses and the gloss for each of the sense (you will see that synsets might contain related words, but you only need the senses that contain the word of interest or derivates thereof; this should be specified in the function). Print the output for "bat".
# 

# In[8]:


import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

def get_senses_and_glosses(word):
    """
    Returns {synset_name: gloss} for all synsets that contain `word`
    (or a derivate of it) as one of their lemma names.
    """
    senses = {}
    for synset in wn.synsets(word):
        lemma_names = [l.name() for l in synset.lemmas()]
        if any(word in ln for ln in lemma_names):
            senses[synset.name()] = synset.definition()
    return senses

# Test with "bat"
bat_senses = get_senses_and_glosses("bat")
print(f"Senses for 'bat' ({len(bat_senses)} total):")
for sense, gloss in bat_senses.items():
    print(f"  {sense:25s}: {gloss}")


# ## Task 2.2: Function to compute sense embeddings [10 points]
# 
# Now that you have a function to extract senses and glosses for a given word, write a function that takes a word and computes embeddings for each of the senses following the method explained in Jurafsky's and Martin's chapter. In this case, no need to calculate at different layers: you should use the last layer only. You should maximally optimise this function like before.
# 
# The output should include the sense, the gloss, and the embedding. Print the function's output when using the word "bank".
# 

# In[9]:


import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer + model for gloss embeddings (same model, separate handle)
sense_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sense_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
sense_model.eval()

def compute_sense_embeddings(word, layer=12):
    """
    For each sense of `word` in WordNet, mean-pool the BERT token embeddings
    of the gloss at `layer` (default: last = 12) to produce a sense embedding.
    Returns a list of dicts: [{sense, gloss, embedding}, ...]

    Following Jurafsky & Martin (2021) ch. 18: the sense embedding is the
    average of the contextualised representations of all tokens in the gloss.
    """
    senses = get_senses_and_glosses(word)
    if not senses:
        return []

    sense_names = list(senses.keys())
    glosses     = list(senses.values())

    # Batch-encode all glosses at once for efficiency
    inputs = sense_tokenizer(
        glosses, return_tensors="pt", padding=True,
        truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = sense_model(**inputs)

    # hidden_states[0] = embedding layer, hidden_states[1..12] = transformer layers
    hidden = outputs.hidden_states[layer]   # (batch, seq_len, 768)
    mask   = inputs["attention_mask"]       # (batch, seq_len)

    results = []
    for i, (sense_name, gloss) in enumerate(zip(sense_names, glosses)):
        n_real = mask[i].sum().item()       # includes [CLS] and [SEP]
        # Mean-pool over real tokens only, excluding [CLS] (pos 0) and [SEP]
        token_embs = hidden[i, 1 : n_real - 1, :]  # (n_tokens, 768)
        sense_emb  = token_embs.mean(dim=0).numpy()
        results.append({"sense": sense_name, "gloss": gloss, "embedding": sense_emb})

    return results

# Test with "bank"
bank_sense_embs = compute_sense_embeddings("bank")
print(f"Sense embeddings for 'bank' ({len(bank_sense_embs)} senses):")
for s in bank_sense_embs:
    print(f"  {s['sense']:20s} shape={s['embedding'].shape}  gloss: {s['gloss']}")


# ## Task 2.3: Compute sense embeddings for the RAW-C stimuli [5 points]
# 
# Now, use the function you defined above to compute sense embeddings for the RAW-C stimuli and pickle dump it too.
# 
# As above, the information that should be there for each word is: the sense, the gloss, the embedding at the last layer. Again, you can think of which structure to use best, but keep in mind that we will have to compare these to the CWE calculated in task 1, so it is good to think of a similar structure that is easily comparable.
# 
# Make sure that the number of stimuli matches the number of stimuli in the final RAW-C dataset.

# In[10]:


from tqdm import tqdm

unique_lemmas = sorted({s["lemma"] for s in stimuli})
print(f"Computing sense embeddings for {len(unique_lemmas)} unique lemmas...")

sense_data = {}   # {lemma: [{sense, gloss, embedding}, ...]}
no_senses  = []

for lemma in tqdm(unique_lemmas):
    result = compute_sense_embeddings(lemma)
    sense_data[lemma] = result
    if not result:
        no_senses.append(lemma)

print(f"\nDone. Words with senses found : {len(sense_data) - len(no_senses)}")
if no_senses:
    print(f"Words with NO WordNet senses : {no_senses}")

# Verify count matches RAW-C stimuli
print(f"Matches RAW-C stimuli count (112): {len(sense_data) == 112}")

# Pickle dump to Drive
SENSE_PATH = f"{SAVE_DIR}/sense_data.pkl"
with open(SENSE_PATH, "wb") as f:
    pickle.dump(sense_data, f)
print(f"Saved to {SENSE_PATH}")

# Reload and verify
with open(SENSE_PATH, "rb") as f:
    sense_data_loaded = pickle.load(f)
print(f"Reloaded successfully — total words: {len(sense_data_loaded)}")


# # Task 3: Compute and explore similarity between homonym CWEs and sense embeddings [35 points]
# 
# You now have the homonym CWEs computed in task 1, and the sense embeddings computed in task 2. The next step is to calculate cosine similarities between each CWE for each homonym (at the selected layer!) and each sense embedding for that homonym.
# 
# For instance, say for the word "bat" with meaning M1_a, you have its CWE at the static layer and at layers 4, 8, 12 and 7 senses: here, you will end up with 16 cosine similarities (take each CWE and compute its similarity to each of the sense embeddings). We then want to see which sense meaning is the closest to each CWE, and do some qualitative explorations with that.
# 
# ## Task 3.1: Compute the cosine similarity between all the CWEs and the sense embeddings [8 points]
# 
# This task is not trivial with regards to how much information you have and how to structure the data (this is why it's also important to think of data structures in the earlier parts of the assignment), so take some time to think how to best breakdown this task. Test each step/function if you have multiple. Pickle dump your final output so that it is easily retrievable for later. At the end, print an example of the entry "bank".
# 
# For cosine similarity, the cdist function from scipy.spatial.distance seems the most efficient, but you are free to use any of your liking (hint: pay attention to the shape of your embeddings and to similarity vs distance. You will need the similarity).

# In[11]:


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

LAYERS = ["static", "4", "8", "12"]

sim_data = []   # one dict per (lemma, meaning_id, layer, sense)

for entry in cwe_data:
    lemma      = entry["lemma"]
    meaning_id = entry["meaning_id"]
    senses     = sense_data.get(lemma, [])
    if not senses:
        continue

    sense_names = [s["sense"] for s in senses]
    sense_embs  = np.stack([s["embedding"] for s in senses])  # (n_senses, 768)

    for layer in LAYERS:
        cwe = entry["embeddings"][layer].reshape(1, -1)        # (1, 768)
        # cdist gives cosine *distance*; similarity = 1 - distance
        dists = cdist(cwe, sense_embs, metric="cosine")[0]     # (n_senses,)
        sims  = 1 - dists

        for sense_name, sim in zip(sense_names, sims):
            sim_data.append({
                "lemma":      lemma,
                "lexeme":     entry["lexeme"],
                "meaning_id": meaning_id,
                "layer":      layer,
                "sense":      sense_name,
                "similarity": float(sim),
            })

print(f"Total similarity entries: {len(sim_data)}")

# Pickle dump
SIM_PATH = f"{SAVE_DIR}/sim_data.pkl"
with open(SIM_PATH, "wb") as f:
    pickle.dump(sim_data, f)
print(f"Saved to {SIM_PATH}")

# --- Print 'bank' entries as a check ---
bank_sim = [e for e in sim_data if e["lemma"] == "bank"]
bank_df  = pd.DataFrame(bank_sim)
print(f"\n=== All similarity entries for 'bank' ({len(bank_sim)} rows) ===")
print(bank_df.to_string(index=False))


# ## Task 3.2: Quantitative and qualitative explorations the relationship between homonym embeddings and dominant senses
# 
# Now, we can look into how the CWEs in different meanings and layers relate to the different senses of a homonym. We'll focus on the dominant sense in WordNet, see below for more details. This section includes both code blocks and reflection questions.

# ### Dominant senses in WordNet and top senses across layers (focus on static layer) [8 points]
# 
# Embeddings at the static layer do not take into account context, so intuitively they should capture the 'average' meaning, maybe the most common/dominant. We can test this by looking at the most similar sense and seeing if that matches that most common/dominant sense in the synset.
# 
# Keep in mind that synsets mark more common/dominant senses with numbering: so n.01 will be the most common noun; v.01 the most common verb, etc. If that is not available, the most common meaning will be the next number (e.g., n.02). You have to take that into account when you extract the top sense, so first extract information about which are the most dominant senses for each word across all the parts of speech: for example, "bat" might have as its two most common senses bat.n.01 and bat.v.02 (because v.01 might not be available; this is just an example). Some words might only have one part of speech in their synset, some more. Print your results.

# In[12]:


def get_dominant_senses(word):
    """
    For each part of speech (n, v, a, r, ...), return the synset with the
    lowest ordinal number that contains `word` as a lemma.
    These are WordNet's most frequent/dominant senses.
    """
    senses = get_senses_and_glosses(word)
    if not senses:
        return {}
    pos_best = {}
    for synset_name in senses:
        # format: word.pos.number  e.g. bank.n.01
        parts = synset_name.split(".")
        pos, num = parts[1], int(parts[2])
        if pos not in pos_best or num < pos_best[pos][1]:
            pos_best[pos] = (synset_name, num)
    return {pos: sn for pos, (sn, _) in pos_best.items()}

# Compute dominant senses for all words in the dataset
dominant_senses = {}
for lemma in {s["lemma"] for s in stimuli}:
    dominant_senses[lemma] = get_dominant_senses(lemma)

print("=== Dominant senses (first 15 words) ===")
for lemma in sorted(dominant_senses.keys())[:15]:
    print(f"  {lemma:15s}: {dominant_senses[lemma]}")


# Then, extract the top similarity of homonyms to the senses at all the layers you have available. While we are interested in the static layer for checking dominant senses, it is also interesting to look into other layers to see whether adding context will refine the captured meaning.
# 

# In[13]:


# Build a DataFrame from sim_data
sim_df = pd.DataFrame(sim_data)

# For each (lemma, meaning_id, layer) find the sense with the highest similarity
top_sim_df = (
    sim_df
    .sort_values("similarity", ascending=False)
    .groupby(["lemma", "meaning_id", "layer"], sort=False)
    .first()
    .reset_index()
    [["lemma", "lexeme", "meaning_id", "layer", "sense", "similarity"]]
)

print(f"Top-similarity dataframe shape: {top_sim_df.shape}")
print("\nFirst 12 rows:")
print(top_sim_df.head(12).to_string(index=False))


# Let's check an example from our results.
# 
# Out of all the similarities of 'bank' to all its senses at all the layers, which one is the highest? Print your results for that entry and reflect below.

# In[14]:


# Top sense for 'bank' across all layers and meanings
# Use a categorical order so layers sort as: static, 4, 8, 12
layer_order = pd.CategoricalDtype(["static", "4", "8", "12"], ordered=True)
bank_top = (
    top_sim_df[top_sim_df["lemma"] == "bank"]
    .copy()
    .assign(layer=lambda df: df["layer"].astype(layer_order))
    .sort_values(["layer", "meaning_id"])
)
print("=== Top sense per (meaning_id, layer) for 'bank' ===")
print(bank_top.to_string(index=False))

# Single highest-similarity entry across all layers and meanings
best = sim_df[sim_df["lemma"] == "bank"].nlargest(1, "similarity").iloc[0]
print(f"\n=== Highest similarity overall for 'bank' ===")
print(f"  meaning_id : {best['meaning_id']}")
print(f"  layer      : {best['layer']}")
print(f"  sense      : {best['sense']}")
print(f"  similarity : {best['similarity']:.4f}")
print(f"  gloss      : {sense_data['bank'][[s['sense'] for s in sense_data['bank']].index(best['sense'])]['gloss']}")


# ### Does the static layer capture the most dominant meaning, according to WordNet (and according to you)? [2 point]
# 
# No, the static layer does not capture the most dominant meaning of "bank" according to WordNet. At the static layer, the top sense for all four meanings is `bank.n.09`, which is a relatively obscure sense (a flight manoeuvre). The dominant WordNet senses are `bank.n.01` (sloping land beside water) and `bank.v.01` (tip laterally). This makes sense because static embeddings are context-free: they represent a single averaged vector for the word form "banked", which does not necessarily align with WordNet's frequency-based ordering. Intuitively, "banked" as a word form is indeed commonly associated with flight/plane manoeuvres, so the static embedding may be picking up on the morphological form rather than the lemma's most common meaning.
# 
# ### Across other layers and meanings, which layer seems to capture the meaning of bank across meanings best, and why do you make this conclusion? [2 points]
# 
# Layer 12 appears to capture the meaning of "bank" across meanings best. At layer 12, the M2 meanings (M2_a: "He banked the money", M2_b: "He banked the cash") correctly map to `depository_financial_institution.n.01` with the highest similarities overall (0.60 and 0.57), while the M1 meanings (M1_a: "He banked the plane", M1_b: "He banked the helicopter") map to `bank.n.10`, a different sense. This shows that layer 12 differentiates between the two distinct meanings of "bank" across contexts, whereas layers 4 and 8 assign the same top sense (`depository_financial_institution.n.01`) to all four contexts indiscriminately. Layer 12 has had the most contextual processing and thus best captures meaning in context.

# ### Checking matches and mismatches with the dominant sense [5 points]
# 
# Now, let's quantitatively check if the static layer actually captures the most dominant sense (any POS). You should end up with two data structures: matches (when the most similar sense is one of the dominant senses) and mismatches (when the most similar sense is not one of the dominant sense). Do that also for the other layers to compare. Print the percentage of matches and mismatches per layer.
# 
# 

# In[15]:


# For each layer, count how many words' top sense IS one of their dominant senses
match_rows = []

for layer in LAYERS:
    layer_top = top_sim_df[top_sim_df["layer"] == layer]
    matches = mismatches = 0
    for _, row in layer_top.iterrows():
        dom = list(dominant_senses.get(row["lemma"], {}).values())
        if row["sense"] in dom:
            matches += 1
        else:
            mismatches += 1
    total = matches + mismatches
    match_rows.append({
        "layer":       layer,
        "matches":     matches,
        "mismatches":  mismatches,
        "% match":     round(100 * matches   / total, 1) if total else 0,
        "% mismatch":  round(100 * mismatches / total, 1) if total else 0,
    })

match_df = pd.DataFrame(match_rows)
print("=== Matches and mismatches per layer ===")
print(match_df.to_string(index=False))


# Now, print the matches and mismatches for the static layer only.

# In[16]:


static_top = top_sim_df[top_sim_df["layer"] == "static"].copy()

static_matches    = []
static_mismatches = []

for _, row in static_top.iterrows():
    dom = list(dominant_senses.get(row["lemma"], {}).values())
    entry = {
        "lemma":            row["lemma"],
        "meaning_id":       row["meaning_id"],
        "top_sense":        row["sense"],
        "similarity":       round(row["similarity"], 4),
        "dominant_senses":  dom,
    }
    if row["sense"] in dom:
        static_matches.append(entry)
    else:
        static_mismatches.append(entry)

print(f"=== STATIC LAYER — MATCHES ({len(static_matches)}) ===")
for e in static_matches:
    print(f"  {e['lemma']:15s} ({e['meaning_id']})  top={e['top_sense']}  dominant={e['dominant_senses']}")

print(f"\n=== STATIC LAYER — MISMATCHES ({len(static_mismatches)}) ===")
for e in static_mismatches:
    print(f"  {e['lemma']:15s} ({e['meaning_id']})  top={e['top_sense']}  dominant={e['dominant_senses']}")


# ### Do BERT's static embeddings capture the most dominant sense in WordNet? [2 point]
# 
# Only to a limited extent. The static layer achieves a match rate of just 22.3%, meaning that for roughly 1 in 5 (word, meaning) pairs, the most similar sense embedding corresponds to one of WordNet's dominant senses. The remaining 77.7% are mismatches. This is not surprising: BERT's static (layer 0) embeddings are non-contextualised word-piece embeddings that represent a compressed average of all the contexts the word appears in during pre-training. This "average" does not necessarily align with WordNet's frequency-based sense ordering, which is derived from manually annotated corpus data (SemCor).
# 
# ### Do the percentages of matches and mismatches throughout the layers make sense to you or is it different than what you expected? [2 points]
# 
# The trend across layers does make sense. The match percentages increase from 22.3% (static) to 21.7% (layer 4), 27.5% (layer 8), and 31.2% (layer 12). Higher layers in BERT carry more contextual information, and the glosses that define dominant senses tend to describe the most prototypical uses of a word, which are also the uses most likely to appear in typical sentences. The slight dip from static (22.3%) to layer 4 (21.7%) is somewhat surprising but could be attributed to noise, since the difference is small. What is perhaps unexpected is that even at layer 12, the match rate is only 31.2%, suggesting that BERT's contextualised representations and WordNet's sense inventory capture meaning in fundamentally different ways.
# 
# ### For the **static layer**, are there any words that seem to particularly deviate from the dominant meaning? If so, which and why could that be? [3 points]
# 
# Several words deviate notably from the dominant meaning at the static layer:
# 
# - **"run"**: the static embedding maps to `footrace.n.01` instead of the dominant `run.n.01` (an act of running) or `run.v.01`. This could be because in BERT's training corpus, "run" co-occurs frequently with sports and racing contexts.
# - **"date"**: the static embedding picks `date.n.02` (a particular day) instead of `date.n.01` (the present, as in "to date"). The second sense may be more frequent in written text (news articles, documents), causing the static embedding to lean towards it.
# - **"newspaper"**: maps to `newspaper.n.04` (a daily paper) instead of `newspaper.n.01` (a publication). This is a fine-grained distinction where both senses are closely related, and the static embedding picks a non-dominant but semantically near sense.
# - **"film"**: matches `film.v.02` (to record on film) instead of `movie.n.01` or `film.v.01`. This suggests that the word form "film" in BERT's embedding space gravitates toward the action of filming rather than the artifact.
# - **"pupil"**: maps to `schoolchild.n.01` instead of `student.n.01`. Both denote learners, but they are listed as separate synsets in WordNet, so this is a near-miss that reveals the granularity issue of WordNet's sense inventory.
# 
# ### Do you think the corpus on which BERT is trained might reflect different meaning dominance than for WordNet's senses? If so/not, why? [3 points]
# 
# Yes, BERT's training corpus very likely reflects different meaning dominance than WordNet's sense ordering. There are several reasons for this:
# 
# 1. **Corpus composition**: BERT was trained on BooksCorpus and English Wikipedia, which overrepresent certain domains (encyclopaedic knowledge, fiction) relative to the balanced corpora (e.g., SemCor/Brown Corpus) used to establish WordNet's sense frequencies. For example, "bank" in Wikipedia likely skews heavily toward the financial institution meaning, while WordNet's dominant sense is the riverbank.
# 
# 2. **Temporal differences**: WordNet's sense frequencies were established decades ago and reflect language usage patterns from that period. BERT's training data reflects more modern usage, where certain senses may have become more or less prominent (e.g., "cell" as in "cell phone" is now far more common than when WordNet was first compiled).
# 
# 3. **Representation method**: BERT's static embeddings collapse all contextual occurrences into a single vector via subword tokenisation, so the "dominant" meaning in BERT space is an implicit average weighted by corpus frequency. WordNet's sense ordering is based on explicit human annotation of sense frequency. These two approaches to capturing dominance are fundamentally different and need not converge.

# # Task 4: Partially replicate Trott & Bergen's experiment [20 points]
# 
# Now comes the time to partially replicate the RAW-C experiment, by seeing whether different layers of BERT capture meanings more or less similarly to humans. At the end you will have to wrap up with a brief comment on which layer seems to capture meanings best and how that connects to explorations in the previous section.
# 
# ## Task 4.1: Create a dataframe with cosine similarities between sentences at different layers [7 points]
# 
# You should now use the embeddings at the different layers that you computed to calculate similarities between each context: M1a, M1b, M2a, M2b. You will have to have all combinations, so for each string in the RAW-C dataframe, you'll have: M1a vs M1b, M1a vs M2a, M1a vs M2b, M1b vs M2a, M1b vs M2b, M2a vs M2b.
# 
# Bear in mind that your final dataframe should include: the word, the string as it appears in the sentence, cosine similarity at layers 4, layer 8, layer 12, the version being compared (is it M1a vs M1b or M1a vs M2a?) and the mean relatadness given by humans (hint: the repo you cloned will come useful here, both in terms of code and data). Print the head of the dataframe to check everything is in order, and check also that the number of stimuli match with your number across the assignment (starting from task 1).

# In[17]:


from scipy.spatial.distance import cosine as cosine_dist

rawc_df = pd.read_csv("raw-c/data/processed/raw-c.csv")

# Build fast lookup: (lexeme, meaning_id) -> layer embeddings
cwe_lookup = {(e["lexeme"], e["meaning_id"]): e["embeddings"] for e in cwe_data}

rows   = []
missing = 0

for _, row in rawc_df.iterrows():
    k1 = (row["string"], row["v1"])
    k2 = (row["string"], row["v2"])

    if k1 not in cwe_lookup or k2 not in cwe_lookup:
        missing += 1
        continue

    emb1 = cwe_lookup[k1]
    emb2 = cwe_lookup[k2]

    new_row = {
        "word":             row["word"],
        "string":           row["string"],
        "version":          row["version"],
        "same":             row["same"],
        "mean_relatedness": row["mean_relatedness"],
    }
    for layer in ["4", "8", "12"]:
        new_row[f"cosine_layer_{layer}"] = 1 - cosine_dist(emb1[layer], emb2[layer])

    rows.append(new_row)

results_df = pd.DataFrame(rows)

print(f"Rows in dataframe : {len(results_df)}  (missing: {missing})")
print(f"Unique words      : {results_df['word'].nunique()}  (expected 112)")
print("\nHead:")
print(results_df.head(6).to_string(index=False))

# Save
RESULTS_PATH = f"{SAVE_DIR}/results_df.pkl"
results_df.to_pickle(RESULTS_PATH)
print(f"\nSaved to {RESULTS_PATH}")


# ## Task 4.2: Correlate with human judgements and visualise [8 points]
# 
# First, correlate the cosine similarities at the different layers to the mean human relatedness judgements. Use the same correlation metric used by Trott & Bergen.

# In[18]:


from scipy.stats import spearmanr

# Trott & Bergen (2021) use Spearman rank correlation
print("=== Spearman correlations: BERT layer vs. mean human relatedness ===\n")
for layer in [4, 8, 12]:
    r, p = spearmanr(results_df[f"cosine_layer_{layer}"], results_df["mean_relatedness"])
    print(f"  Layer {layer:2d}:  r = {r:+.4f},  p = {p:.3e}")


# Next, visualise your results. You want to see the correlation between BERT embeddings and human judgements per layer, but what would also be interesting is to include the meaning contrasts (such as M1_a_M1_b, etc), so that we can see how those play out per layer.

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sort versions consistently so the legend is readable
version_order = ["M1_a_M1_b", "M2_a_M2_b", "M1_a_M2_a", "M1_a_M2_b", "M1_b_M2_a", "M1_b_M2_b"]
palette = sns.color_palette("tab10", n_colors=len(version_order))

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

for ax, layer in zip(axes, [4, 8, 12]):
    r, _ = spearmanr(results_df[f"cosine_layer_{layer}"], results_df["mean_relatedness"])
    sns.scatterplot(
        data=results_df,
        x=f"cosine_layer_{layer}",
        y="mean_relatedness",
        hue="version",
        hue_order=version_order,
        palette=palette,
        alpha=0.55,
        s=35,
        ax=ax,
        legend=(layer == 12),
    )
    ax.set_title(f"Layer {layer}  (r = {r:.3f})", fontsize=12)
    ax.set_xlabel("Cosine similarity (BERT)", fontsize=10)
    ax.set_ylabel("Mean human relatedness" if layer == 4 else "", fontsize=10)
    ax.grid(alpha=0.3)

# Move legend outside the last panel
if axes[-1].get_legend():
    axes[-1].get_legend().set_title("Comparison")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

plt.suptitle("BERT contextual similarity vs. human relatedness across layers (RAW-C)", fontsize=13)
plt.tight_layout()

plot_path = f"{SAVE_DIR}/task4_correlations.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to {plot_path}")
plt.show()


# ### Reflect on the correlations and on the visualisations. What can you observe and infer in terms of which layer(s) might be capturing meaning best? Is there one way to determine that (i.e., what does 'capturing meanings' mean?)? Contrast and compare the layers. [5 points]
# 
# The Spearman correlations show a clear, monotonically increasing trend: layer 4 (r = 0.49), layer 8 (r = 0.63), and layer 12 (r = 0.67). All correlations are highly significant (p < 10^-41), but layer 12 provides the strongest alignment with human relatedness judgements. This is consistent with the finding from Task 3, where layer 12 also showed the highest match rate with dominant senses (31.2%).
# 
# From the visualisations, several patterns emerge:
# - **Same-meaning pairs** (M1_a_M1_b, M2_a_M2_b) tend to cluster in the upper-right region (high cosine similarity, high human relatedness), while **different-meaning pairs** (M1_a_M2_a, etc.) cluster in the lower-left. This separation becomes more pronounced at higher layers, confirming that later layers better capture the distinction between same and different meanings.
# - At **layer 4**, the data points are more compressed along the x-axis (cosine similarities are generally higher and closer together), meaning that layer 4 does not strongly differentiate between same-meaning and different-meaning uses. This aligns with the lower correlation (r = 0.49).
# - At **layers 8 and 12**, the spread increases: different-meaning pairs shift to lower cosine values, and same-meaning pairs maintain high cosine values, resulting in better separation and higher correlations.
# 
# However, there is no single way to determine what "capturing meanings best" means. From a **human alignment** perspective, layer 12 is best, as it most closely mirrors human judgements. But from a **sense disambiguation** perspective (Task 3), even layer 12 only matches WordNet's dominant sense 31% of the time. This discrepancy arises because human relatedness is a graded, continuous measure (how related do two uses *feel*?), while sense matching is a discrete, categorical decision (is it *this* exact sense?). BERT's contextualised representations seem to capture graded meaning similarity better than categorical sense identity.
# 
# It is also worth noting that BERT systematically **underestimates** the similarity of same-meaning pairs and **overestimates** the similarity of different-meaning pairs relative to humans, as Trott and Bergen (2021) also found. This suggests that while higher layers capture contextual meaning differences, BERT's geometric notion of cosine distance does not perfectly map onto human intuitions about meaning relatedness. Human judgements likely incorporate world knowledge, pragmatic inference, and experiential grounding that a text-only language model cannot fully capture.
# 
# 
# 
# 
