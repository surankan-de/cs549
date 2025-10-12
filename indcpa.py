# indcpa.py - Paper's MINE with Original NN Architecture
"""
Exact MINE implementation from paper (Equation 1) with your original NN architecture
"""
from systems import *
import os
import random
import argparse
from typing import Tuple, List
import numpy as np
from Crypto.Cipher import AES, DES
from Crypto.Random import get_random_bytes
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# ALGORITHM 2: IND-CPA BCE Classification
# -------------------------

def generate_non_uniform_plaintext(size: int) -> bytes:
    """Non-uniform plaintext: all zeros"""
    return b'\x00' * size

def generate_uniform_plaintext(size: int) -> bytes:
    """Uniform random plaintext"""
    return get_random_bytes(size)

def make_indcpa_dataset(n_samples: int, msg_len: int, keys, cipher_name: str):
    """Algorithm 2: BCE Classification for IND-CPA"""
    
    if cipher_name.startswith('RSA'):
        ct_len = 256
    elif cipher_name in ['AES CTR', 'AES CTR Reduced']:
        ct_len = msg_len + 16
    elif cipher_name == 'DES NonDet':
        ct_len = 16
    else:
        ct_len = msg_len
    
    X = np.zeros((2 * n_samples, ct_len), dtype=np.float32)
    y = np.zeros((2 * n_samples,), dtype=np.int64)
    
    # Y0: Encrypt non-uniform plaintexts (label 0)
    for i in range(n_samples):
        m = generate_non_uniform_plaintext(msg_len)
        c = encrypt_variant(cipher_name, keys, m)
        
        c_arr = np.frombuffer(c, dtype=np.uint8).astype(np.float32) / 255.0
        if len(c_arr) >= ct_len:
            X[i] = c_arr[:ct_len]
        else:
            X[i, :len(c_arr)] = c_arr
        y[i] = 0
    
    # Y1: Encrypt uniform plaintexts (label 1)
    for i in range(n_samples):
        m = generate_uniform_plaintext(msg_len)
        c = encrypt_variant(cipher_name, keys, m)
        
        c_arr = np.frombuffer(c, dtype=np.uint8).astype(np.float32) / 255.0
        if len(c_arr) >= ct_len:
            X[n_samples + i] = c_arr[:ct_len]
        else:
            X[n_samples + i, :len(c_arr)] = c_arr
        y[n_samples + i] = 1
    
    return X, y

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLClassifier:
    """
    Drop-in replacement for neural IND-CPA classifier.
    Supports Logistic Regression or Random Forest backends.
    """
    def __init__(self, input_dim=None, model_type="logistic", **kwargs):
        """
        Args:
            input_dim: unused (for API compatibility)
            model_type: "logistic" or "rf"
            kwargs: passed directly to sklearn model
        """
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 8),
                n_jobs=-1,
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                max_iter=kwargs.get("max_iter", 500),
                solver=kwargs.get("solver", "lbfgs"),
                n_jobs=-1
            )

    def fit(self, X, y, epochs=None, batch_size=None, lr=None, device=None):
        """Mimics train_classifier() signature; ignores DL args."""
        self.model.fit(X, y)

    def evaluate(self, X, y):
        """Returns (accuracy, predictions)."""
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        return acc, preds

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

class MLMIEstimator:
    """
    Classical Mutual Information estimator (non-neural).
    Supports 'knn' (mutual_info_regression) and 'kde' (histogram) modes.
    """
    def __init__(self, method="knn", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.mi_value = None

    def estimate(self, X_plain, X_cipher):
        """
        Estimate mutual information between plaintext and ciphertext samples.
        Args:
            X_plain: np.ndarray (N × d_p)
            X_cipher: np.ndarray (N × d_c)
        Returns:
            float: estimated MI
        """
        if self.method == "knn":
            # average over plaintext dimensions
            mis = []
            for i in range(X_plain.shape[1]):
                mi = mutual_info_regression(X_cipher, X_plain[:, i], discrete_features=False)
                mis.append(np.mean(mi))
            self.mi_value = float(np.mean(mis))

        elif self.method == "kde":
            # flatten to 1D histograms for simplicity
            x = X_plain.flatten()
            y = X_cipher.flatten()
            c_xy = np.histogram2d(x, y, bins=self.kwargs.get("bins", 32))[0]
            self.mi_value = float(mutual_info_score(None, None, contingency=c_xy))

        else:
            raise ValueError(f"Unknown MI method: {self.method}")

        return self.mi_value

    def get_value(self):
        """Return the last computed MI."""
        return self.mi_value


class ClassifierNet(nn.Module):
    """Your original classifier architecture"""
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2)
        )
    
    def forward(self, x):
        return self.net(x)

from sklearn.model_selection import train_test_split

def train_classifier(X: np.ndarray, y: np.ndarray, device: str = None,
                    epochs: int = 20, batch_size: int = 256, lr: float = 1e-3) -> Tuple[nn.Module, float]:
    """Your original training procedure"""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X_train ,X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    X_t1 = torch.tensor(X_test, dtype=torch.float32)
    y_t1 = torch.tensor(y_test, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TinyClassifier(input_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_t1.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = (preds == y_t1.cpu()).float().mean()
    
    return model, acc


# -------------------------
# PAPER'S MINE Implementation (Equation 1)
# -------------------------
class TinyClassifier(nn.Module):
    """
    Very small MLP for IND-CPA tests.
    ~8k parameters, works well for small inputs (like 64–256 bits).
    """
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2)
        )
    
    def forward(self, x):
        return self.net(x)
class TinyMINE(nn.Module):
    """
    Lightweight MINE estimator — 2 layers × 64 neurons.
    Suitable for small ciphers and CPU-only edge devices.
    """
    def __init__(self, dim_m: int, dim_c: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_m + dim_c, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, m, c):
        x = torch.cat([m, c], dim=1)
        return self.net(x)
    
class MineNet(nn.Module):
    """
    Exact paper architecture:
    - 2 hidden layers × 100 units each
    - ELU activations
    """
    def __init__(self, dim_m: int, dim_c: int, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_m + dim_c, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, m, c):
        x = torch.cat([m, c], dim=1)
        return self.net(x)


class MINE:
    """
    Donsker–Varadhan estimator with Choi & Lee stabilizer term
    (Equation 1 from the paper).

    Iφ(X;Y) = E_{P(X,Y)}[Fφ] − log E_{P(X)P(Y)}[e^{Fφ}]
              − 0.1 * (log E_{P(X)P(Y)}[e^{Fφ}])²
    """
    def __init__(self, dim_m: int, dim_c: int, device: str = None, lr: float = 1e-4):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = TinyMINE(dim_m, dim_c, hidden=100).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)

    def estimate_mi(self, M: np.ndarray, C: np.ndarray,
                    batch_size: int = 512, epochs: int = 200, verbose: bool = False):
        M_t = torch.tensor(M, dtype=torch.float32).to(self.device)
        C_t = torch.tensor(C, dtype=torch.float32).to(self.device)
        N = M_t.shape[0]
        history = []

        for epoch in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                mb = M_t[idx]
                cb = C_t[idx]

                # Negative samples (marginal shuffle)
                perm_neg = torch.randperm(N)[:mb.size(0)]
                c_neg = C_t[perm_neg]

                T_pos = self.net(mb, cb)
                T_neg = self.net(mb, c_neg)

                # Stable log(mean(exp()))
                lse = torch.logsumexp(T_neg, dim=0)
                log_mean_exp = lse - torch.log(torch.tensor(T_neg.size(0), device=self.device))

                Ep = T_pos.mean()
                stabilizer = 0.1 * (log_mean_exp ** 2)
                loss = -(Ep - log_mean_exp - stabilizer)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

            with torch.no_grad():
                perm_eval = torch.randperm(N)
                T_pos_all = self.net(M_t, C_t)
                T_neg_all = self.net(M_t, C_t[perm_eval])
                Ep_all = T_pos_all.mean().item()
                lse_all = torch.logsumexp(T_neg_all, dim=0).item() - np.log(N)
                mi_est = Ep_all - lse_all
                history.append(float(mi_est))
                if verbose and (epoch % max(1, epochs // 10) == 0):
                    print(f"[MINE] epoch {epoch+1}/{epochs}  I(M;C) = {mi_est:.4f} nats")

        return history[-1], history

def bytes_list_to_matrix(byte_list: List[bytes], target_len: int) -> np.ndarray:
    N = len(byte_list)
    mat = np.zeros((N, target_len), dtype=np.float32)
    for i, b in enumerate(byte_list):
        arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
        arr = (arr / 127.5) - 1.0             # 🔧 map [0,255] → [-1,1]
        L = min(target_len, arr.shape[0])
        mat[i, :L] = arr[:L]
    return mat




def run_experiment(
    cipher_name: str = 'AES',
    keys=None,
    block_bytes: int = None,
    indcpa_samples: int = 10000,
    indcpa_epochs: int = 15,
    mine_samples: int = 10000,
    mine_epochs: int = 1000,
    device: str = None
):
    print(f"\n{'='*60}")
    print(f"Testing: {cipher_name}")
    print(f"{'='*60}")
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if keys is None:
        keys = generate_keys()
    
    if cipher_name.startswith('RSA'):
        msg_len = 100
    elif cipher_name.startswith('AES'):
        msg_len = 16
    elif cipher_name.startswith('DES'):
        msg_len = 8
    else:
        msg_len = 16
    
    # Reset counters
    if cipher_name == 'RSA OAEP Reused':
        import systems
        systems._rsa_oaep_reused_counter = 0
    elif cipher_name == 'AES CTR Reduced':
        import systems
        systems._aes_ctr_reduced_counter = 0
    
    # IND-CPA Classification
    print("\n[Algorithm 2] Generating IND-CPA dataset...")
    X_cls, y_cls = make_indcpa_dataset(
        n_samples=indcpa_samples,
        msg_len=msg_len,
        keys=keys,
        cipher_name=cipher_name
    )
    
    print("Training classifier...")
    cls_model, acc = train_classifier(
        X_cls, y_cls,
        device=device,
        epochs=indcpa_epochs,
        lr=1e-3
    )
    # model = MLClassifier(model_type="logistic")  # or "logistic"
    # model.fit(X_cls, y_cls)
    # acc, _ = model.evaluate(X_test, y_test)
    
    advantage = 2.0 * (acc - 0.5)
    print(f"✓ Classifier accuracy: {acc:.4f}")
    print(f"✓ IND-CPA advantage ε = {advantage:.4f}")

    # MINE MI Estimation
    print("\n[Algorithm 1] Generating samples for MINE...")
    
    if cipher_name.startswith('RSA'):
        ct_len = 256
    elif cipher_name in ['AES CTR', 'AES CTR Reduced']:
        ct_len = msg_len + 16
    elif cipher_name == 'DES NonDet':
        ct_len = 16
    else:
        ct_len = msg_len
    
    Ms = []
    Cs = []
    for _ in range(mine_samples):
        m = get_random_bytes(msg_len)
        c = encrypt_variant(cipher_name, keys, m)
        Ms.append(m)
        Cs.append(c)

    M_mat = bytes_list_to_matrix(Ms, msg_len)
    C_mat = bytes_list_to_matrix(Cs, ct_len)

    print("Training MINE estimator...")
    mine = MINE(dim_m=msg_len, dim_c=ct_len, device=device, lr=1e-4)
    mi_value, history = mine.estimate_mi(
        M_mat, C_mat,
        batch_size=256,
        epochs=mine_epochs,
        verbose=True
    )
    
    mi_bits = mi_value / np.log(2)
    print(f"✓ Estimated I(M;C) ≈ {mi_value:.6f} nats ({mi_bits:.2f} bits)")

    return {
        "adv_accuracy": float(acc),
        "advantage": float(advantage),
        "mi_estimate": float(mi_value),
        "mine_history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indcpa-samples", type=int, default=10000)
    parser.add_argument("--mine-samples", type=int, default=10000,
                       help="Paper uses 100,000 for training")
    parser.add_argument("--indcpa-epochs", type=int, default=15)
    parser.add_argument("--mine-epochs", type=int, default=100,
                       help="Paper uses 1000 epochs")
    args = parser.parse_args()

    keys = generate_keys()
    systems_to_test =[
        # Diagnostic / toy systems
        'No Encryption',
        'One-Time Pad',
        'Constant XOR',
        'Toy Fixed XOR',
        'Toy Substitution',
        'Toy Permutation',
        'Toy 1-Round Feistel',
        'AES CTR Fixed Nonce',

        # Real ciphers
        'DES',
        'DES NonDet',
        'AES ECB',
        'AES CTR',
        'AES CTR Reduced',
        'RSA Plain',
        'RSA OAEP',
        'RSA OAEP Reused',

        # Additional toy and semi-weak systems
        'Toy Caesar',
        'Toy Repeating XOR',
        'Toy Byte Rotate',
        'Toy Mask HighNibble',
        'Toy LFSR Stream',
        'Toy 2-Round Feistel',
        'Semi Reduced Feistel',
        'Semi Partial Mask',
        'Semi Truncated AES',
        'Semi Nonce Mix',
        'Semi LFSR Long',
        'Semi Key Rotation'
    ]

    results_summary = []
    for sysname in systems_to_test:
        try:
            res = run_experiment(
                cipher_name=sysname,
                keys=keys,
                indcpa_samples=args.indcpa_samples,
                mine_samples=args.mine_samples,
                indcpa_epochs=args.indcpa_epochs,
                mine_epochs=args.mine_epochs,
                device=args.device
            )
            results_summary.append((sysname, res['adv_accuracy'], res['mi_estimate']))
        except Exception as e:
            print(f"[!] Error testing {sysname}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'System':<25} | {'IND-CPA Acc':<12} | {'MI Estimate':<12}")
    print("-"*70)
    for name, acc, mi in results_summary:
        print(f"{name:<25} | {acc:>11.3f} | {mi:>11.4f}")