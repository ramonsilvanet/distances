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

labels = pd.read_csv(os.path.join(DATA_DIR, "label_all_new.csv"), names=['imagenet_img_id', 'label'], header=0)
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


tam_i = len(nouns_vectors)
tam_j = len(labels_vectors)

print("Liberando memoria...")

dataset = []
cache  = {}
for i in range(0, tam_i):

    img_id = nouns['vqa_img_id'][i]

    if img_id in cache:
        continue

    print("Processando", img_id)
    dataset = []
    
    cache[img_id] = 1
    
    for j in range(0, tam_j):
        distance = euclidean_distances([nouns_vectors[i]], [labels_vectors[j]])       
        if distance[0] > 1:
            continue        
        print([labels['imagenet_img_id'][j], nouns['question_id'][i], labels['label'][j], nouns['noun'][i], distance[0,0]])
        dataset.append([labels['imagenet_img_id'][j], int(nouns['question_id'][i]), labels['label'][j], nouns['noun'][i], distance[0,0]])

    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(DATA_DIR, "bboxes_imagenet", 'dist_calc_new.csv'), mode='a', index=0, header=0)


print("Finalizado")