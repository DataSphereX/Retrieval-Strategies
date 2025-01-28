# Sparse Retrieval Strategies

## Overview
In information retrieval, the goal is to find relevant documents from a large corpus based on a user's query. Two common retrieval strategies are TF-IDF (Term Frequency-Inverse Document Frequency) and BM25 (Best Matching 25). This document provides an overview of these strategies and their use cases.

## TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a corpus. It combines two metrics:
- **Term Frequency (TF)**: The number of times a term appears in a document.
- **Inverse Document Frequency (IDF)**: The inverse of the number of documents containing the term, which helps to reduce the weight of common terms.

### Strengths
- Simple to implement and computationally efficient.
- Works well for smaller datasets and straightforward retrieval tasks.

### Weaknesses
- Does not handle term frequency saturation well, leading to diminishing returns for very frequent terms.
- May not perform as well on larger and more complex datasets.

## BM25 (Best Matching 25)
BM25 is an advanced probabilistic retrieval model that improves upon TF-IDF by incorporating term frequency saturation and document length normalization. It is part of the Okapi BM25 family of ranking functions.

### Strengths
- More sophisticated handling of term frequency saturation.
- Generally provides better retrieval performance for larger and more complex datasets.
- Incorporates document length normalization, making it more robust for varying document lengths.

### Weaknesses
- Slightly more complex to implement and tune compared to TF-IDF.

## Example Usage
Here is a simple example comparing TF-IDF and BM25 for a given query:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly.",
    "A quick brown dog outpaces a quick fox.",
    "Lazy dogs are not quick to jump over."
]

# TF-IDF Retrieval
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
query = "quick fox"
query_vector = tfidf_vectorizer.transform([query])
cosine_similarities = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

print("TF-IDF Scores:")
for i, score in enumerate(cosine_similarities):
    print(f"Document {i+1}: {score}")

# BM25 Retrieval
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = query.split(" ")
bm25_scores = bm25.get_scores(tokenized_query)

print("\nBM25 Scores:")
for i, score in enumerate(bm25_scores):
    print(f"Document {i+1}: {score}")
