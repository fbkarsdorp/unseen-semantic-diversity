
import glob
import tqdm
import json

from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import pairwise
from spacy.lang.en import English
from transformers import AutoModel, AutoTokenizer

from get_archer_embeddings import get_embeddings, get_type_embeddings, read_file, scramble_text


def functional_diversity(counts, dm, tau, q):
    dm = np.clip(dm, 0, tau)
    a = np.inner(1 - dm / tau, counts)
    counts = counts[a != 0]
    a = a[a != 0]
    v = counts / a
    n = counts.sum()
    if q == 1:
        return np.exp(np.sum(-v * a / n * np.log(a / n)))
    return np.sum(v * (a / n)**q) ** (1 / (1 - q))


def RaoQ(counts, dm):
    counts = counts / counts.sum()
    Q = 0
    for i in range(dm.shape[0]):
        for j in range(dm.shape[0]):
            Q += dm[i, j] * counts[i] * counts[j]
    return Q


def compute_profile(counts, dm, q_values, tau_values):
    profile = np.zeros((len(tau_values), len(q_values)))
    for i, tau in enumerate(tau_values):
        for j, q in enumerate(q_values):
            profile[i, j] = functional_diversity(counts, dm, tau, q)
    return {
        "profile": profile,
        "tau_values": tau_values,
        "q_values": q_values,
    }


def compute_q_profile(counts, dm, q_min, q_max, resolution=30):
    non_diag = np.invert(np.eye(dm.shape[0], dtype=bool))
    q_values = np.linspace(q_min, q_max, resolution)
    tau_values = np.min(dm[non_diag]), RaoQ(counts, dm), np.max(dm)
    return compute_profile(counts, dm, q_values, tau_values)


def compute_tau_profile(counts, dm, tau_min, tau_max, resolution=30):
    tau_values = np.linspace(tau_min, tau_max, resolution)
    q_values = 0, 1, 2
    return compute_profile(counts, dm, q_values, tau_values)


def get_sents(path, p, q=0.2):
    sents = []
    for sent in read_file(path):
        # lowercase
        sent = sent.lower()
        # introduce noise on alpha tokens only
        sent = scramble_text([tok.text for tok in spacy_tok(sent)], p=p, q=q)
        sent = ' '.join(sent)
        sents.append(sent)
    return sents


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',
        default="/home/manjavacasema/code/unseen-semantic-diversity/data/ARCHER3_2/ARCHER_3-2_TXT/*/*txt")
    parser.add_argument('--output',
        default='/home/manjavacasema/data1/unseen-semantic-diversity/profiles.jsonl')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--ft-model')
    args = parser.parse_args()

    # files = "/home/manjavacasema/code/unseen-semantic-diversity/data/ARCHER3_2/ARCHER_3-2_TXT/*/*txt"
    # paths = glob.glob(files)
    paths = glob.glob(args.files)
    print(len(paths))
    spacy_tok = English().tokenizer

    ft_model = tokenizer = model = None
    if args.ft_model:
        # ft_model = KeyedVectors.load("/home/manjavacasema/data1/unseen-semantic-diversity/general_ft/1550-1940.ft")
        ft_model = KeyedVectors.load(args.ft_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh', model_max_length=512)
        model = AutoModel.from_pretrained('emanjavacas/MacBERTh')
        model.to(args.device)

    with open(args.output, 'w+') as output:
        for fpath in tqdm.tqdm(paths, total=len(paths)):
            print(fpath)
            # iterate over p's
            for p in [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75]:
                for it in range(args.iter):
                    fpath = paths[0]
                    p = 0.2
                    sents = get_sents(fpath, p=p)
                    if ft_model is not None:
                        embeddings, n_items = get_type_embeddings(ft_model, sents)
                    else:
                        embeddings, n_items = get_embeddings(
                            model, tokenizer, sents, 
                            batch_size=args.batch_size, device=args.device)
                    keys, vecs = zip(*embeddings.items())
                    counts = np.array([n_items[key] for key in keys])

                    # get dm
                    dm = pairwise.cosine_distances(vecs)
                    tau_profile = compute_tau_profile(counts, dm, 0, 1)
                    q_profile = compute_q_profile(counts, dm, 0, 3)

                    row = json.dumps({
                        'q_profile': q_profile, 
                        'tau_profile': tau_profile, 
                        'p': p,
                        'iteration': it,
                        'file': fpath}, 
                        cls=NumpyArrayEncoder)

                    output.write(row + '\n')