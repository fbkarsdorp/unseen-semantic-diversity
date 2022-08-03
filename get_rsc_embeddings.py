
import json
import numpy as np
from gensim.models import KeyedVectors
import collections


counts_tok = collections.Counter()
counts_lem = collections.Counter()
with open('./data/Royal_Society_Corpus_v2.0.2_final.json') as f:
    for line in f:
        line = json.loads(line.strip())
        langs = sorted(line['langs'], key=lambda it: it['prob'], reverse=True)
        if langs[0]['lang'] != 'en' or langs[0]['prob'] < 0.85:
            continue
        counts_lem.update(line['lemma'])
        counts_tok.update(line['token'])


def count_uppercase(w):
    return len([ch for ch in w if ch.isupper()])


filtered_lem = collections.Counter(
    {w: c for w, c in counts_lem.items() 
        if w.isalpha() and 
        (count_uppercase(w) == 0 or (count_uppercase(w) == len(w))) and
        len(set(w).intersection(set('aeiouAEIOU'))) > 0})

model = KeyedVectors.load('./data/general_ft/1550-1940.ft')

def get_vecs(words):
    vecs, keys = [], []
    for lemma in set(words):
        vec = model[lemma]
        vec = vec / np.linalg.norm(vec)
        vecs.append(vec)
        keys.append(lemma)
    return vecs, keys

vecs, keys = get_vecs(list(filtered_lem))
with open('rsc.embs.lemmas.npy', 'wb') as f:
    np.savez(f, vecs=np.array(vecs), keys=np.array(keys))

# hapax = [w for w, c in filtered_lem.most_common() if c == 1]
# filtered = [w for w in hapax if w.isalpha()]
# random.shuffle(filtered)
# filtered[:50]
# len(filtered)
