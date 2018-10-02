import os, sys, gensim, logging
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = os.environ["DATA_DIR"]


nouns = pd.read_csv(os.path.join(DATA_DIR, "vqa_nouns.csv"), names=['vqa_img_id', 'question_id', 'noun'], header=0)

#nouns_vectors = []

nouns_list = {}

for index, row in nouns.iterrows():
    img_id = row["vqa_img_id"]
    noun = str(row["vqa_img_id"]).lower()
    if img_id in nouns_list:
        nouns_list[img_id].append(noun)
    else:
        nouns_list[img_id] = [noun]

print("VQA Imagens", len(nouns_list))

labels = pd.read_csv(os.path.join(DATA_DIR, "label_all_new.csv"), names=['imagenet_img_id', 'label'], header=0)
labels_list = {}

for index, row in labels.iterrows():
    img_id = row["imagenet_img_id"]
    label = str(row["label"]).lower()
    if img_id in labels_list:
        labels_list[img_id].append(label)
    else:
        labels_list[img_id] = [label]

print("IMAGENET Imagens", len(labels_list))


print("Finalizado")