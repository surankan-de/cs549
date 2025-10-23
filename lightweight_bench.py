#!/usr/bin/env python3
"""Lightweight model benchmark for IND-CPA classification.

Runs multiple scikit-learn models on IND-CPA datasets and reports accuracy,
training time, inference time, and approximate model size on disk.
"""
import time
import pickle
import os
import numpy as np
from indcpa import make_indcpa_dataset, train_test_split, systems_to_test
from systems import generate_keys

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


MODELS = {
    'logistic_l2': LogisticRegression(max_iter=1000),
    'logistic_l1': LogisticRegression(penalty='l1', solver='saga', max_iter=1000),
    'linear_svc': LinearSVC(max_iter=2000),
    'dtree': DecisionTreeClassifier(max_depth=6),
    'rf_small': RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1),
    'gb_small': GradientBoostingClassifier(n_estimators=100, max_depth=3),
    'knn3': KNeighborsClassifier(n_neighbors=3),
    'gaussian_nb': GaussianNB()
}


def flatten_X_from_make(X):
    # X from make_indcpa_dataset is floats in [0,1], shape (N, dim)
    return X.reshape(X.shape[0], -1)


def sizeof(obj) -> int:
    # approximate by pickle size
    return len(pickle.dumps(obj))


def run_bench(cipher_name='Toy Substitution', n_samples=5000, msg_len=16):
    print(f"\nRunning benchmark for {cipher_name} with {n_samples} samples per class")
    keys = generate_keys()
    X, y = make_indcpa_dataset(n_samples=n_samples, msg_len=msg_len, keys=keys, cipher_name=cipher_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    Xtr = flatten_X_from_make(X_train)
    Xte = flatten_X_from_make(X_test)

    results = []
    for name, model in MODELS.items():
        m = model
        start = time.time()
        try:
            m.fit(Xtr, y_train)
        except Exception as e:
            print(f"Model {name} failed to fit: {e}")
            continue
        train_time = time.time() - start

        start = time.time()
        preds = m.predict(Xte)
        inf_time = time.time() - start
        acc = accuracy_score(y_test, preds)

        size = sizeof(m)

        print(f"{name:12s} | acc={acc:.4f} | train_s={train_time:.3f} | inf_s={inf_time:.3f} | size_bytes={size}")
        results.append((name, acc, train_time, inf_time, size))

    return results


if __name__ == '__main__':
    for cipher in systems_to_test:
        run_bench(cipher_name=cipher, n_samples=2000, msg_len=16)
