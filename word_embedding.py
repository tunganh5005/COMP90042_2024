from gensim.utils import simple_preprocess

from preprocess import preprocess_sentence

from gensim.models import Word2Vec

import json

model = Word2Vec.load("my_word2vec_model.model")


def train():

    # Opening JSON file 
    evidence_file = open('data/evidence.json', 'rb')

    # Load JSON content
    documents = list(json.load(evidence_file).values())

    sentences = [preprocess_sentence(doc) for doc in documents]

    # Train the Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Save the model for later use
    model.save("my_word2vec_model.model")

    
def find_similar(word):
    # Load the model (if it's not loaded)
    model = Word2Vec.load("my_word2vec_model.model")

    # Find words similar to 'language'
    similar_words = model.wv.most_similar(word, topn=5)
    
    return similar_words

def embed(word):
    return model.wv[word]

if __name__ == '__main__':

    # model = Word2Vec.load("my_word2vec_model.model")
    # train()
    # print(find_similar('electricity'))

    print(embed('new_york'))
    # print(len(model.wv))


