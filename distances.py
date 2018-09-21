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

labels = pd.read_csv(os.path.join(DATA_DIR, "label_all.csv"), names=['imagenet_img_id', 'label'], header=0)
nouns = pd.read_csv(os.path.join(DATA_DIR, "vqa_nouns.csv"), names=['vqa_img_id', 'question_id', 'noun'], header=0)

print("Processando distancias")

nouns_vectors = []

for index, row in nouns.iterrows():
    noun = str(row["noun"]).lower()
    if noun in model:        
        nouns_vectors.append( model[noun] )
    else:
        print(noun, "nao existe")

labels_vectors = []
for k, v in labels.iterrows():    
    label = str(v["label"]).lower()
    if label in model:
        labels_vectors.append(model[label])
    else:
        print(label, "nao existe")
		
distances = euclidean_distances(nouns_vectors, labels_vectors)

print("Distances", distances.shape)

print("Liberando memoria...")
del nouns_vectors[:]
del labels_vectors[:]

tam_i, tam_j = distances.shape

dataset = []

for i in range(0, tam_i):
    print("Processando", nouns['vqa_img_id'][i])
    dataset = []
    for j in range(0, tam_j):        
        print([labels['imagenet_img_id'][j], nouns['vqa_img_id'][i], nouns['question_id'][i], labels['label'][j], nouns['noun'][i], distances[i,j]])
        dataset.append([labels['imagenet_img_id'][j], nouns['vqa_img_id'][i], nouns['question_id'][i], labels['label'][j], nouns['noun'][i], distances[i,j]])

    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(DATA_DIR, 'dist_calc.csv'), mode='a', index=0, header=0)


print("Finalizado")