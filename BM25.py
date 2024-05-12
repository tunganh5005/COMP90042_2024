import json
from collections import defaultdict
import math
import pickle

import nltk

from gensim.utils import simple_preprocess

from gensim.models import Word2Vec

import re

from scipy.spatial.distance import cosine

from nltk.corpus import wordnet
nltk.download('words')
nltk.download('wordnet')

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
words = set(nltk.corpus.words.words()) #a list of words provided by NLTK
words = set([ word.lower() for word in words ]) #lowercase all the words for better matching


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def remove_special_characters(input_string):
    # Keep letters, numbers, spaces, and underscores
    result_string = re.sub(r'[^\w\s]', '', input_string)  # \w matches letters, numbers, and underscores
    return result_string

# def preprocess(text):
#     """Preprocess the text: tokenize and lower case."""
#     # Simple tokenization and case normalization
#     return remove_special_characters(text.lower()).split()


def preprocess(text):
    """Preprocess the text to keep multi-word entities intact while handling other splits correctly."""
    # Remove non-alphanumeric characters except spaces, dashes, and apostrophes, which might be part of proper nouns
    cleaned_text = re.sub(r'[^\w\s\'.]', ' ', text)
    
    cleaned_text = cleaned_text.replace('. ', '')

    cleaned_text = cleaned_text.replace(' .', '')


    # Identify and temporarily replace multi-word named entities with underscores to keep them as single tokens
    # Regex explanation:
    # \b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b matches sequences of words that start with an uppercase letter
    pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')
    matches = pattern.findall(cleaned_text)
    temp_map = {}
    for i, match in enumerate(matches):
        temp_key = f"__ENTITY{i}__"
        temp_map[temp_key] = match
        cleaned_text = cleaned_text.replace(match, temp_key)

    # Split the text normally
    tokens = cleaned_text.split()

    # Replace the placeholders with original named entities, converting spaces to underscores
    final_tokens = []
    for token in tokens:
        if token in temp_map:
            # Replace spaces with underscores for the named entity
            final_tokens.append(temp_map[token].lower().replace(" ", "_"))
        else:
            final_tokens.append(lemmatize(token).lower())

    return final_tokens


def calculate_doc_freq(documents):
    """Calculate the document frequency of each term in the document set."""
    doc_freq = defaultdict(int)
    # Iterate over each document and count terms
    for doc_id, text in documents.items():
        # Preprocess and find unique terms in the document
        terms = set(preprocess(text))
        # Increment the document frequency for each unique term
        for term in terms:
            doc_freq[term] += 1
    return doc_freq

def compute_tf(token, doc):
    """Compute term frequency for a token in a document."""
    return doc.count(token)

def compute_idf(token, doc_freq, total_docs):
    """Compute inverse document frequency for a token."""
    return math.log((total_docs - doc_freq.get(token, 0) + 0.5) / (doc_freq.get(token, 0) + 0.5) + 1)

def calculate_avgdl(documents):
    """Calculate the average document length for a collection of documents."""
    total_length = 0
    num_documents = len(documents)
    
    for document in documents.values():
        # Tokenize the document and count words
        tokens = preprocess(document)
        total_length += len(tokens)
    
    # Calculate average document length
    avgdl = total_length / num_documents if num_documents > 0 else 0
    return avgdl



def semantic_tf(query_word, doc_tokens, model, threshold=0.8):
    query_vec = model.wv[query_word] if query_word in model.wv else None
    if query_vec is None:
        return 0
    weighted_tf = 0
    for word in doc_tokens:
        if word in model.wv:
            sim = 1 - cosine(query_vec, model.wv[word])
            if sim > threshold:
                weighted_tf += sim  # Weighted by cosine similarity
    return weighted_tf


def bm25(query, document, doc_freq, total_docs, avgdl, k1=1.2, b=1):
    """Compute BM25 score for a single document given a query."""
    # Preprocess documents
    doc_tokens = preprocess(document)
    query_tokens = preprocess(query)
    
    # Document length
    doc_len = len(doc_tokens)
    
    # Calculate BM25
    score = 0
    for token in query_tokens:
        if token in doc_tokens:
            # print(token)
            tf = compute_tf(token, doc_tokens)
            idf = compute_idf(token, doc_freq, total_docs)
            term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl))))
            if '_' in token:
                term_score = term_score * 3
            score += term_score
    
    return score


def bm25_word2vec(query, document, doc_freq, total_docs, avgdl, model, k1=1.8, b=0.5, threshold=0.9):
    """Compute BM25 score for a single document given a query using semantic similarities."""
    # Preprocess documents
    doc_tokens = preprocess(document)
    query_tokens = preprocess(query)
    
    # Document length
    doc_len = len(doc_tokens)
    
    # Calculate BM25
    score = 0
    for token in query_tokens:
        tf = semantic_tf(token, doc_tokens, model, threshold)
        if tf > 0:  # Only consider if there's any semantic match
            idf = compute_idf(token, doc_freq, total_docs)
            term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl))))
            score += term_score
    return score

def retrieve_documents(data, query, doc_freq, total_docs, avgdl, threshold):
    """Retrieve documents that have a BM25 score above a certain threshold for a given query."""
    
    # Scoring documents
    results = {}
    for doc_id, doc in data.items():
        score = bm25(doc, query, doc_freq, total_docs, avgdl)
        if score > threshold:
            results[doc_id] = score
            print(f'{doc_id}: {doc}')
    
    return results


def retrieve_documents_modified(data, query, doc_freq, total_docs, avgdl, model, threshold=30):
    """Retrieve documents that have a BM25 score above a certain threshold for a given query."""
    
    # Scoring documents
    results = {}
    for doc_id, doc in data.items():
        score = bm25_word2vec(doc, query, doc_freq, total_docs, avgdl, model)
        if score > threshold:
            results[doc_id] = score
            print(f'{doc_id}: {doc}')
    
    return results


if __name__ == "__main__":

    # Opening JSON file 
    evidence_file = open('data/evidence.json', 'rb')

    # Load JSON content
    documents = json.load(evidence_file)

    # print(documents)

    try:
        with open('doc_freq.pickle', 'rb') as f:
            doc_freq = pickle.load(f)
    except:
        # Calculate document frequency
        doc_freq = calculate_doc_freq(documents)
        with open('doc_freq.pickle', 'wb') as f:
            pickle.dump(doc_freq, f)


    try:
        with open('avg_dl.pickle', 'rb') as f:
            avgdl = pickle.load(f)
    except:
        # Calculate document frequency
        avgdl = calculate_avgdl(documents)
        with open('avg_dl.pickle', 'wb') as f:
            pickle.dump(avgdl, f)


    # avgdl = avgdl - 10


    # Calculate total number of documents
    total_docs = len(documents.keys())

    model = Word2Vec.load("my_word2vec_model.model")

    query = 'Global warming is driving major melting on the surface of Greenland\u2019s glaciers and is speeding up their travel into the sea.\u201d'

    # doc = 'Climate change caused by human activities that emit greenhouse gases into the air is expected to affect the frequency of extreme weather events such as drought, extreme temperatures, flooding, high winds, and severe storms.'

    print(preprocess(query))

    doc = 'This could lead to changing, and for all emissions scenarios more unpredictable, weather patterns around the world, less frost days, more extreme events (droughts and storm or flood disasters), and warmer sea temperatures and melting glaciers causing sea levels to rise.'

    print(preprocess(doc))

    score = bm25(query, doc, doc_freq, total_docs, avgdl)

    print(score)
    
    # print(score)

    # score = bm25_word2vec(query, doc, doc_freq, total_docs, avgdl, model)

    # retrieve_docs = retrieve_documents_modified(documents, query, doc_freq, total_docs, avgdl, model)
    # print(retrieve_docs)

    # retrieved_docs = retrieve_documents(documents, query, doc_freq, total_docs, avgdl, 15)

    # print(retrieved_docs)

    # dev_claim_file = open('data/dev-claims.json', 'rb')

    # # Load JSON content
    # dev_claim = json.load(dev_claim_file)

    # for claim in dev_claim.keys():
        