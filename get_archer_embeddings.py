
import tqdm
import random
import string
import os
import pathlib
import collections

from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import numpy as np
from spacy.lang.en import English
from transformers import AutoModel, AutoTokenizer
import torch


def random_char():
    return random.choice(string.ascii_lowercase)


def scramble_word(word, p=0.2):
    return ''.join(random_char() if (c.isalpha() and random.random() < p) else c for c in word)


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


def read_files(*paths, spacy_tok, p, q=0.2):
    for path in paths:
        sents = []
        for sent in read_file(path):
            # lowercase
            sent = sent.lower()
            # introduce noise on alpha tokens only
            sent = scramble_text([tok.text for tok in spacy_tok(sent)], p=p, q=q)
            sents.append(' '.join(sent))
        yield sents


def read_corpus(root, spacy_tok, p, q=0.2):
    sents = []

    for corpus in os.listdir(root):
        if corpus.startswith('.'): continue
        for fn in pathlib.Path(os.path.join(root, corpus)).iterdir():
            sents.extend(list(read_files(fn, spacy_tok=spacy_tok, p=p, q=q)))

    return sents


def get_embeddings(model, tokenizer, sents, batch_size=24, max_length=512, device='cpu'):
    embeddings = collections.defaultdict(torch.tensor)
    n_items = collections.defaultdict(int)
    for b_start in tqdm.tqdm(range(0, len(sents), batch_size), total=len(sents)//batch_size):
        batch = sents[b_start: b_start + batch_size]
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        assert inputs['input_ids'].shape[1] <= max_length
        inputs = {key: t.to(device) for key, t in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        for sent_idx, sent in enumerate(batch):
            for subtoken in get_subtokens(tokenizer, sent):
                token = ''.join(sent[start:end] for _, (start, end) in subtoken)
                if not token.isalpha():
                    continue
                outputs = output['last_hidden_state'][sent_idx]
                idxs = [idx for idx, _ in subtoken]
                if max(idxs) >= max_length:
                    continue
                embedding = outputs[idxs].mean(0).cpu().numpy()
                # cumulative mean
                if n_items[token] == 0:
                    embeddings[token] = embedding
                else:
                    embeddings[token] = (embedding + n_items[token] * embeddings[token]) / (n_items[token] + 1)
                n_items[token] += 1

    return embeddings, n_items


def get_type_embeddings(model, sents):
    n_items = collections.Counter(token.lower() for sent in sents for token in sent.split() if token.isalpha())
    embeddings = {w: model[w.lower()] for w in n_items}
    return embeddings, n_items


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="./data/ARCHER3_2/ARCHER_3-2_TXT/")
    parser.add_argument('--embeddings-path', required=True)
    parser.add_argument('--p', required=True, type=float)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # batch processing of sentences
    spacy_tok = English().tokenizer
    tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh', model_max_length=512)
    model = AutoModel.from_pretrained('emanjavacas/MacBERTh')
    model.to(args.device)

    sents = read_corpus(args.root, spacy_tok, args.p)

    # sents = read_corpus("./data/ARCHER3_2/ARCHER_3-2_TXT/", spacy_tok, 0.2)
    # embeddings, n_items = get_embeddings(model, tokenizer, sents[:100])

    embeddings, n_items = get_embeddings(
        model, tokenizer, sents, batch_size=args.batch_size, device=args.device)

    keys, vecs = zip(*embeddings.items())
    counts = [n_items[key] for key in keys]
    with open(args.embeddings_path, 'wb') as f:
        np.savez(f, vecs=np.array(vecs), keys=np.array(keys), counts=np.array(counts))
