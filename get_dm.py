
import random
import string
import os
import pathlib
import collections

from nltk.tokenize import sent_tokenize
from spacy.lang.en import English
from transformers import AutoModel, AutoTokenizer
import torch


def random_char():
    return random.choice(string.ascii_lowercase)


def scramble_word(word, p=0.2):
    return ''.join(random_char() if (c.isalpha() and random.random() < p) else c for c in word )


def scramble_text(words, p=0.1, q=0.1):
    return [word if random.random() > p else scramble_word(word, p=q) for word in words]


def read_file(fpath):
    with open(fpath, encoding="latin-1") as f:
        line = next(f)
        while line.startswith('<'):
            line = next(f)
            # skip metadata
            continue
        text = (line + f.read())
        text = ' '.join(text.split())
        return sent_tokenize(text)


def get_subtokens(tokenizer, sent):
    encoded = tokenizer.encode_plus(sent, return_offsets_mapping=True)
    mapping = encoded['offset_mapping']
    indices = list(enumerate(mapping))
    indices = indices[1:-1] # drop first item
    idx, (start, end) = indices.pop(0)
    subtokens = [(idx, (start, end))]
    output = []
    for new in indices:
        _, (n_start, n_end) = new
        if n_start != end:
            output.append(subtokens)
            subtokens = [new]
        else:
            subtokens.append(new)
        end = n_end
    output.append(subtokens)
    return output


def read_corpus(spacy_tok, p, q=0.2, root="./data/ARCHER3_2/ARCHER_3-2_TXT/"):
    sents = []

    for corpus in os.listdir(root):
        if corpus.startswith('.'): continue
        for fn in pathlib.Path(os.path.join(root, corpus)).iterdir():
            for sent in read_file(fn):
                # lowercase
                sent = sent.lower()
                # introduce noise on alpha tokens only
                sent = scramble_text([tok.text for tok in spacy_tok(sent)], p=p, q=q)
                sents.append(' '.join(sent))

    return sents


# # batch processing of sentences
# spacy_tok = English().tokenizer
# tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')
# model = AutoModel.from_pretrained('emanjavacas/MacBERTh')
# batch_size = 24

# sents = read_corpus(spacy_tok, 0.2)
# embeddings = collections.defaultdict(torch.tensor)
# n_items = collections.defaultdict(int)

# for b_start in range(0, len(sents), batch_size):
#     batch = sents[b_start: b_start + batch_size]
#     inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         output = model(**inputs)
#     for sent_idx, sent in enumerate(batch):
#         for subtoken in get_subtokens(tokenizer, sent):
#             token = ''.join(sent[start:end] for _, (start, end) in subtoken)
#             if not token.isalpha():
#                 continue
#             idxs = [idx for idx, _ in subtoken]
#             embedding = output['last_hidden_state'][sent_idx][idxs].mean().cpu().numpy()
#             # cumulative mean
#             if n_items[token] == 0:
#                 embeddings[token] = embedding
#             else:
#                 embeddings[token] = (embedding + n_items[token] * embeddings[token]) / (n_items[token] + 1)

