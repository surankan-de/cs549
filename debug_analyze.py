#!/usr/bin/env python3
"""Debug helper: analyze IND-CPA dataset ciphertext uniqueness and train/test overlap
"""
import numpy as np
from collections import Counter
from indcpa import make_indcpa_dataset, train_test_split, generate_uniform_plaintext
from systems import generate_keys
from sklearn.linear_model import LogisticRegression


def to_bytes_from_Xrow(row: np.ndarray) -> bytes:
    # X rows are floats in [0,1] (from make_indcpa_dataset). Map back to uint8 bytes.
    arr = np.clip((row * 255.0).round(), 0, 255).astype(np.uint8)
    return bytes(arr.tobytes())


def analyze(cipher_name='Toy Substitution', n_samples=2000, msg_len=16):
    print(f"Analyzing cipher='{cipher_name}' n_samples={n_samples} msg_len={msg_len}")
    keys = generate_keys()
    X, y = make_indcpa_dataset(n_samples=n_samples, msg_len=msg_len, keys=keys, cipher_name=cipher_name)

    # Convert rows back to bytes
    cts = [to_bytes_from_Xrow(r) for r in X]

    print("Total samples:", len(cts))
    print("Unique ciphertexts overall:", len(set(cts)))

    labels = list(map(int, y.tolist())) if hasattr(y, 'tolist') else list(map(int, list(y)))
    for lbl in sorted(set(labels)):
        lbl_cts = [c for c, l in zip(cts, labels) if l == lbl]
        print(f"label {lbl}: samples={len(lbl_cts)} unique={len(set(lbl_cts))}")
        sample_hex = [c[:24].hex() for c in lbl_cts[:5]]
        print("examples (first 5, truncated hex):", sample_hex)

    # Check if any ciphertext appears in both labels
    set0 = set(c for c, l in zip(cts, labels) if l == 0)
    set1 = set(c for c, l in zip(cts, labels) if l == 1)
    inter = set0 & set1
    print("Ciphertexts present in both labels:", len(inter))

    # Check train/test split overlap like in train_classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_cts = set(to_bytes_from_Xrow(r) for r in X_train)
    test_cts = set(to_bytes_from_Xrow(r) for r in X_test)
    overlap = train_cts & test_cts
    print("Train/test ciphertext overlap count:", len(overlap))

    # Quick logistic regression sanity check on flat features
    clf = LogisticRegression(max_iter=500)
    flatX = np.array([np.frombuffer(c, dtype=np.uint8).astype(np.float32) for c in cts])
    flatX = flatX.reshape(len(flatX), -1)
    clf.fit(flatX, labels)
    acc = clf.score(flatX, labels)
    print("Logistic regression accuracy on entire dataset:", acc)


if __name__ == '__main__':
    analyze(cipher_name='Toy Substitution', n_samples=2000, msg_len=16)
