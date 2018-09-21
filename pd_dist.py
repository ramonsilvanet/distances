import os, sys
import pandas as pd


DATA_DIR = os.environ["DATA_DIR"]

labels = pd.read_csv(os.path.join(DATA_DIR, "label_all_5.csv"), names=['imagenet_img_id', 'label'], header=0)
nouns = pd.read_csv(os.path.join(DATA_DIR, "vqa_nouns_5.csv"), names=['vqa_img_id', 'imagenet_img_id', 'noun'], header=0)

print( labels['imagenet_img_id'][0] )       