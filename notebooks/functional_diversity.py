import random
import string

import numpy as np


def random_char():
    return random.choice(string.ascii_lowercase)


def scramble_word(word, p=0.2):
    return ''.join(c if random.random() > p else random_char() for c in word)


def scramble_text(words, p=0.1, q=0.1):
    return [word if random.random() > p else scramble_word(word, p=q) for word in words]


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
