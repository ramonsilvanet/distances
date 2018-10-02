import os, sys, gensim, logging, mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

DATA_DIR = os.environ["DATA_DIR"]
VECTORS_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
WORDS_FILE = os.path.join(DATA_DIR, 'questions-words.txt')

DISTANCES_FOLDER = os.path.join(DATA_DIR, 'distances')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
model.accuracy(WORDS_FILE)

nouns_vectors = []
nouns_vectors.append( model["dog"] )

labels_vectors = []
labels_vectors.append(model["item"])

distances = euclidean_distances(nouns_vectors, labels_vectors)

print(distances)