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
	noun = str(row["noun"]).lower()
	if noun in model:
		nouns_vectors.append(model[noun])

labels_vectors = []
for k, v in labels.iterrows():
	label = str(v["label"]).lower()
	if label in model:
		labels_vectors.append( model[label] )

distances = euclidean_distances(nouns_vectors, labels_vectors)

	
print("Inserindo registros")

cache = {}

for i in range(0, len(distances)-1):
	for j in range(0, len(distances[0] - 1 )):
		dist = float(distances[i,j])		
		
		if(dist < 3.0):
			#insert_distances(int(nouns.loc[i]["question_id"]),  labels.loc[j]['image'], dist)
			question_id = int(nouns.loc[i]["question_id"])
			label = labels.loc[j]['image']
			key = "{}#{}".format(question_id, label)
			if key in cache:
				if dist < cache[key]:
					cache[key] = dist
			else:
				cache[key] = dist
	if i % 10000 == 0:
		print(i, "registros processados")

		
print("Tamamho", len(cache))
		

cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')
cursor = cnx.cursor()


cursor.execute("ALTER TABLE distances DISABLE KEYS")

def insert_distances(question_id, img, distance):
	update_url = "INSERT INTO distances (question_id, imagenet_img_id, distance) values (%s, %s, %s)"
	data = (question_id, img, distance)
	cursor.execute(update_url, data)

i = 0

for k, v in cache.items():
    row = k.split("#")
    insert_distances(int(row[0]),row[1],float(v))
    if i % 10000 == 0:
        print(i, "resgistros inseridos")
    i = i + 1

cursor.execute("ALTER TABLE distances ENABLE KEYS")			

cursor.close()
cnx.close()

print("finalizado")
