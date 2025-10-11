# main.py
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader,random_split

from data_gen import *
from utils import PTCTDataset
from models import *
from train_mine import train_mine, mutual_info_mine
from train_classifier import train_classifier
mc = secrets.token_bytes(100)
def build_dataset(cipher, n=20000, pt_len=128, ct_len=128, num_classes=2):
    """Generate plaintext/ciphertext dataset depending on cipher"""
    if cipher in ["aes-ecb", "aes-ctr", "aes-ctr-reduced", "des","xor","caesar","sbox"]:
        # multiclass plaintexts
        pts, labels = generate_plaintexts_classes(n, length=pt_len, num_classes=num_classes)
    else:
        pts = generate_plaintexts_random(n, length=pt_len)
        labels = np.zeros(n, dtype=int)

    if cipher == "aes-ecb":
        key, cts = aes_ecb_encrypt_batch(pts)
    elif cipher == "aes-ctr":
        key, cts = aes_ctr_encrypt_batch(pts, reduced_counter=False)
    elif cipher == "aes-ctr-reduced":
        key, cts = aes_ctr_encrypt_batch(pts, reduced_counter=True, counter_bits=8)
    elif cipher == "des":
        key, cts = des_ecb_encrypt_batch(pts)
        ct_len = 8
    elif cipher == "rsa":
        key, cts = rsa_encrypt_batch(pts, key_size=1024)
        ct_len = max(len(c) for c in cts)
    elif cipher == "xor":
        key,cts = xor_encrypt_batch_multikey(pts ,labels,num_classes)
    elif cipher == "shift":
        key,cts = shift_encrypt_batch_multikey(pts,labels,num_classes)
    elif cipher == "rotate":
        key,cts = rotate_encrypt_batch_multikey(pts,labels,num_classes)
    elif cipher == "affine":
        key,cts = affine_encrypt_batch_multikey(pts,labels,num_classes)
    elif cipher == "vignere":
        key,cts = vigenere_like_multikey_encrypt_batch(pts,labels,num_classes)
    elif cipher == "vignere-auto":
        key,cts = vigenere_autokey_encrypt_batch(pts,labels,num_classes)
    elif cipher =="caesar":
        key,cts = caesar_encrypt_batch(pts)
    elif cipher=="sbox":
        key,cts = substitution_encrypt_batch(pts,labels,num_classes)
    else:
        raise ValueError("Unknown cipher")

    ds = PTCTDataset(pts, cts, labels=labels, pt_len=pt_len, ct_len=ct_len)
    return ds, len(set(labels)) if labels is not None else 2, pt_len, ct_len

def run_experiment(cipher, device="cpu", epochs=5, batch_size=256):
    print(f"=== Running experiment for {cipher} ===")

    # Build dataset
    ds, num_classes, pt_len, ct_len = build_dataset(cipher)

    # Split dataset: 80% train, 20% test
    dataset_size = len(ds)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Determine dimensions from a batch
    pt, ct, labels = next(iter(train_loader))
    pt_dim, ct_dim = pt.shape[1], ct.shape[1]

    # 1. Train MINE (MI estimate) on full dataset
    mine = MINE(pt_dim=pt_dim, ct_dim=ct_dim, hidden=128).to(device)
    print("\n[+] Training MINE estimator ...")
    mi_history = train_mine(mine, train_ds, device=device, epochs=epochs, batch_size=batch_size)
    final_mi = mi_history[-1]
    print(f"[{cipher}] Final MI estimate: {final_mi:.6f}\n")

    # 2. Train classifier on train set
    if args.model == "linear":
        clf = LinearClassifier(pt_dim, ct_dim, num_classes=num_classes).to(device)
    # elif args.model == "nonlinear":
    #     clf = NonLinearClassifier(pt_len, ct_len,num_classes=num_classes).to(device)
    else:
        clf = Classifier(pt_dim, ct_dim, hidden=256, num_classes=num_classes).to(device)
    print("[+] Training classifier (IND-CPA test) ...")
    clf = train_classifier(clf, train_ds, device=device, epochs=epochs, batch_size=batch_size)

    # 3. Evaluate classifier on test set
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pt, ct, labels in test_loader:
            pt, ct, labels = pt.to(device), ct.to(device), labels.to(device)
            logits = clf(pt, ct)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"[{cipher}] Test classifier accuracy: {acc:.4f}\n")

    return final_mi, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cipher", type=str, default="xor",
                        choices=["aes-ecb","affine","vignere","vignere-auto"
                                  "aes-ctr", "aes-ctr-reduced", "des","xor","caesar","sbox","shift","rotate"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "linear","nonlinear"],
                    help="Which classifier to use: mlp or linear")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(args.cipher, device=device, epochs=args.epochs, batch_size=args.batch_size)
