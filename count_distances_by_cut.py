import os, sys
import gensim
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import mysql.connector

DATA_DIR = "/home/rsilva/cefet/datasets/"

VECTORS_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
WORDS_FILE = os.path.join(DATA_DIR, 'questions-words.txt')

DISTANCES_FOLDER = os.path.join(DATA_DIR, 'distances')

model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

labels = pd.read_csv(os.path.join(DATA_DIR, "imagenet_labels.csv"), header=0)
nouns = pd.read_csv(os.path.join(DATA_DIR, "vqa_nouns.csv"), header=0)

nouns_vectors = []

for index, row in nouns.iterrows():
    if row["noun"] in model:        
        nouns_vectors.append( model[row["noun"]] )

labels_vectors = []
for k, v in labels.iterrows():    
    if v["label"] in model:
        labels_vectors.append( model[v["label"]] )

distances = euclidean_distances(nouns_vectors, labels_vectors)

below_baseline = 0
above_baseline = 0

for i in range(0, len(distances)-1):
	for j in range(0, len(distances[0] - 1 )):
		dist = float(distances[i,j])		
		if(dist <= 3.0):
			below_baseline = below_baseline + 1
		else:
			above_baseline = above_baseline + 1
	

print("Abaixo da linha de corte", below_baseline)
print("Acima da linha de corte", above_baseline)