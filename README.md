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
    print(f"Document {i+1}: {score}")```


# Dense Retrieval Examples

This repository contains examples of dense retrieval using BERT and SBERT models. Dense retrieval involves using dense vector representations of queries and documents to find the most relevant documents.

## Overview

Dense retrieval is a technique used in information retrieval where both queries and documents are represented as dense vectors. These vectors are then compared using similarity measures like cosine similarity to retrieve the most relevant documents.

### Models Used

1. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - General-purpose language understanding model.
   - Uses Masked Language Model (MLM) and Next Sentence Prediction (NSP) for training.

2. **SBERT (Sentence-BERT)**:
   - Fine-tuned version of BERT for producing semantically meaningful sentence embeddings.
   - Uses a Siamese network structure for training.

### Comparison

- **BERT**: Requires averaging token embeddings for sentence representation. Suitable for general language understanding tasks.
- **SBERT**: Specifically fine-tuned for sentence embeddings, providing better performance for sentence similarity and dense retrieval tasks.

## Installation

To run the examples, you need to install the following libraries:

```python
pip install transformers sentence-transformers torch scikit-learn
```
Usage
BERT Example
The BERT example demonstrates how to encode queries and documents using a pre-trained BERT model and compute similarity scores to find the most relevant document.


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

query = "What is dense retrieval?"
documents = [
    "Dense retrieval uses dense vector representations.",
    "Sparse retrieval uses term-based representations.",
    "BERT is a transformer model."
]

query_vector_bert = encode_bert(query, tokenizer, model)
document_vectors_bert = [encode_bert(doc, tokenizer, model) for doc in documents]

def get_similarity(query_vector, document_vectors):
    similarities = cosine_similarity([query_vector], document_vectors)
    return similarities[0]

similarities_bert = get_similarity(query_vector_bert, document_vectors_bert)
most_relevant_doc_index_bert = similarities_bert.argmax()
most_relevant_doc_bert = documents[most_relevant_doc_index_bert]

print(f"BERT - Query: {query}")
print(f"BERT - Most relevant document: {most_relevant_doc_bert}")

SBERT Example
The SBERT example demonstrates how to encode queries and documents using a pre-trained SBERT model and compute similarity scores to find the most relevant document.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained SBERT model
model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_sbert(text, model):
    return model.encode(text)

query = "What is dense retrieval?"
documents = [
    "Dense retrieval uses dense vector representations.",
    "Sparse retrieval uses term-based representations.",
    "BERT is a transformer model."
]

query_vector_sbert = encode_sbert(query, model_sbert)
document_vectors_sbert = [encode_sbert(doc, model_sbert) for doc in documents]

def get_similarity(query_vector, document_vectors):
    similarities = cosine_similarity([query_vector], document_vectors)
    return similarities[0]

similarities_sbert = get_similarity(query_vector_sbert, document_vectors_sbert)
most_relevant_doc_index_sbert = similarities_sbert.argmax()
most_relevant_doc_sbert = documents[most_relevant_doc_index_sbert]

print(f"SBERT - Query: {query}")
print(f"SBERT - Most relevant document: {most_relevant_doc_sbert}")

Conclusion
This repository provides examples of how to use BERT and SBERT for dense retrieval tasks. SBERT is generally better suited for sentence similarity and dense retrieval due to its fine-tuning for these tasks. BERT can also be used but may require additional processing to achieve similar performance.
