
import glob
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import pairwise

forms, lemmas = [], []
for f in glob.glob('data/shakespeares-works_TEIsimple_FolgerShakespeare.txt/*txt'):
    with open(f) as inp:
        for line in inp:
            form, lemma = line.rstrip('\n').split('\t')
            forms.append(form)
            lemmas.append(lemma)

model = KeyedVectors.load('./data/1550-1940.w2v')
model.fill_norms()
vecs, keys, missing = [], [], []
for lemma in set(lemmas):
    try:
        vecs.append(model[lemma])
        keys.append(lemma)
    except:
        missing.append(lemma)

len(missing) / len(set(lemmas))
# 0.4283870155221813 missing for forms
# 0.3362848182992068 missing for lemmas

dists = pairwise.euclidean_distances(vecs)
