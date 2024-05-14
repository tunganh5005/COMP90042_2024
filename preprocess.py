import json

import re

import nltk


from nltk.tokenize import word_tokenize

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def tokenize(text):
    return [token.lower() for token in word_tokenize(text)]

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma


def preprocess_sentence(text):
    """Preprocess the text to keep multi-word entities intact while handling other splits correctly."""
    # Remove non-alphanumeric characters except spaces, dashes, and apostrophes, which might be part of proper nouns
    cleaned_text = re.sub(r'[^\w\s\'.-]', ' ', text)
    
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


def prepare_training_set():
    # Opening JSON file 
    evidence_file = open('data/evidence.json', 'rb')

    # Load JSON content
    documents = json.load(evidence_file)


    # Opening JSON file 
    claim_file = open('data/train-claims.json', 'rb')

    # Load JSON content
    claims = json.load(claim_file)

    train_dataset = []

    # print(claims)

    for claim_id in claims.keys():
        query = claims[claim_id]['claim_text']

        label = claims[claim_id]['claim_label']

        for evidence in claims[claim_id]['evidences']:
            document = documents[evidence]

            train_dataset.append({"query": query, "document": document, "label": label})

    with open('train_dataset.json', 'w') as f:
        json.dump(train_dataset, f)

