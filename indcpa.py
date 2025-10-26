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

        # Split into train and test sets (80/20)
        split_idx = int(0.8 * N)
        M_train, M_test = M_t[:split_idx], M_t[split_idx:]
        C_train, C_test = C_t[:split_idx], C_t[split_idx:]

        N_train = M_train.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(N_train)
            for i in range(0, N_train, batch_size):
                idx = perm[i:i+batch_size]
                mb = M_train[idx]
                cb = C_train[idx]

                # Negative samples (marginal shuffle)
                perm_neg = torch.randperm(N_train)[:mb.size(0)]
                c_neg = C_train[perm_neg]

                T_pos = self.net(mb, cb)
                T_neg = self.net(mb, c_neg)

                # Stable log-mean-exp
                lse = torch.logsumexp(T_neg, dim=0)
                log_mean_exp = lse - torch.log(torch.tensor(T_neg.size(0), device=self.device))

                Ep = T_pos.mean()
                stabilizer = 0.1 * (log_mean_exp ** 2)
                loss = -(Ep - log_mean_exp - stabilizer)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

            # Compute MI on full dataset (for history)
            with torch.no_grad():
                perm_eval = torch.randperm(N)
                T_pos_all = self.net(M_t, C_t)
                T_neg_all = self.net(M_t, C_t[perm_eval])
                Ep_all = T_pos_all.mean().item()
                lse_all = torch.logsumexp(T_neg_all, dim=0).item() - np.log(N)
                mi_est = Ep_all - lse_all
                history.append(float(mi_est))

                # Compute MI on test set
                T_pos_test = self.net(M_test, C_test)
                T_neg_test = self.net(M_test, C_test[torch.randperm(C_test.shape[0])])
                Ep_test = T_pos_test.mean().item()
                lse_test = torch.logsumexp(T_neg_test, dim=0).item() - np.log(C_test.shape[0])
                mi_test = Ep_test - lse_test

                if verbose and (epoch % max(1, epochs // 10) == 0):
                    print(f"[MINE] epoch {epoch+1}/{epochs}  I(M;C) = {mi_est:.4f} nats, Test MI = {mi_test:.4f} nats")

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

systems_to_test = [
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

def plot_indcpa_mine_analysis(results_summary):
    """
    Plot IND-CPA accuracy and MINE MI estimates with differential analysis.
    
    Args:
        results_summary: List of tuples (system_name, accuracy, mi_estimate)
    
    Usage:
        results_summary = []
        for sysname in systems_to_test:
            res = run_experiment(...)
            results_summary.append((sysname, res['adv_accuracy'], res['mi_estimate']))
        
        plot_indcpa_mine_analysis(results_summary)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    WEAK_SYSTEMS = [
        'No Encryption', 'Constant XOR', 'Toy Fixed XOR',
        'Toy Substitution', 'Toy Permutation', 'Toy 1-Round Feistel',
        'AES CTR Fixed Nonce', 'Toy Caesar', 'Toy Repeating XOR',
        'Toy Byte Rotate', 'Toy Mask HighNibble', 'Toy LFSR Stream',
        'Toy 2-Round Feistel',
    ]

    SEMI_WEAK_SYSTEMS = [
        'Semi Reduced Feistel', 'Semi Partial Mask', 'Semi Truncated AES',
        'Semi Nonce Mix', 'Semi LFSR Long', 'Semi Key Rotation',
        'DES NonDet', 'AES CTR Reduced',
    ]

    STRONG_SYSTEMS = [
        'DES', 'AES ECB', 'AES CTR', 'RSA Plain', 'RSA OAEP', 'RSA OAEP Reused','One-Time Pad'
    ]
    
    systems, accs, mis = zip(*results_summary)
    systems = list(systems)
    accs = np.array(accs)
    mis = np.array(mis)
    advantages = 2.0 * (accs - 0.5)
    mi_bits = mis / np.log(2)
    
    categories = []
    for sys in systems:
        if sys in WEAK_SYSTEMS:
            categories.append('Weak')
        elif sys in SEMI_WEAK_SYSTEMS:
            categories.append('Semi-Weak')
        elif sys in STRONG_SYSTEMS:
            categories.append('Strong')
        else:
            categories.append('Unknown')
    categories = np.array(categories)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Accuracy by category
    ax1 = fig.add_subplot(gs[0, 0])
    for cat, color in [('Weak', '#ff6b6b'), ('Semi-Weak', '#ffa500'), ('Strong', '#4caf50')]:
        mask = categories == cat
        if mask.any():
            x_pos = np.where(mask)[0]
            ax1.scatter(x_pos, accs[mask], s=100, alpha=0.7, color=color, label=cat, edgecolor='black')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_ylabel('IND-CPA Accuracy', fontweight='bold', fontsize=11)
    ax1.set_title('IND-CPA Classifier Accuracy', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels(systems, rotation=90, fontsize=7)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 2. MI estimates by category
    ax2 = fig.add_subplot(gs[0, 1])
    for cat, color in [('Weak', '#ff6b6b'), ('Semi-Weak', '#ffa500'), ('Strong', '#4caf50')]:
        mask = categories == cat
        if mask.any():
            x_pos = np.where(mask)[0]
            ax2.scatter(x_pos, mi_bits[mask], s=100, alpha=0.7, color=color, label=cat, edgecolor='black')
    ax2.set_ylabel('MI Estimate (bits)', fontweight='bold', fontsize=11)
    ax2.set_title('MINE MI Estimates', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels(systems, rotation=90, fontsize=7)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Accuracy vs MI scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for cat, color in [('Weak', '#ff6b6b'), ('Semi-Weak', '#ffa500'), ('Strong', '#4caf50')]:
        mask = categories == cat
        if mask.any():
            ax3.scatter(mi_bits[mask], accs[mask], s=100, alpha=0.7, color=color, label=cat, edgecolor='black')
            for i in np.where(mask)[0]:
                ax3.annotate(systems[i], (mi_bits[i], accs[i]), fontsize=6, alpha=0.6)
    ax3.set_xlabel('MI Estimate (bits)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('IND-CPA Accuracy', fontweight='bold', fontsize=11)
    ax3.set_title('Accuracy vs MI Correlation', fontweight='bold', fontsize=12)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Advantage distribution
    ax4 = fig.add_subplot(gs[1, 0])
    for cat, color in [('Weak', 'red'), ('Semi-Weak', 'orange'), ('Strong', 'green')]:
        mask = categories == cat
        if mask.any():
            ax4.hist(advantages[mask], bins=15, alpha=0.5, color=color, label=cat, edgecolor='black')
    ax4.set_xlabel('IND-CPA Advantage (2(Acc-0.5))', fontweight='bold', fontsize=10)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax4.set_title('IND-CPA Advantage Distribution', fontweight='bold', fontsize=11)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. MI distribution
    ax5 = fig.add_subplot(gs[1, 1])
    for cat, color in [('Weak', 'red'), ('Semi-Weak', 'orange'), ('Strong', 'green')]:
        mask = categories == cat
        if mask.any():
            ax5.hist(mi_bits[mask], bins=15, alpha=0.5, color=color, label=cat, edgecolor='black')
    ax5.set_xlabel('MI Estimate (bits)', fontweight='bold', fontsize=10)
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax5.set_title('MI Estimate Distribution', fontweight='bold', fontsize=11)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Category comparison bars
    ax6 = fig.add_subplot(gs[1, 2])
    cat_names = ['Weak', 'Semi-Weak', 'Strong']
    cat_colors = ['#ff6b6b', '#ffa500', '#4caf50']
    avg_accs = [accs[categories == cat].mean() if (categories == cat).any() else 0 for cat in cat_names]
    avg_mis = [mi_bits[categories == cat].mean() if (categories == cat).any() else 0 for cat in cat_names]
    
    x = np.arange(len(cat_names))
    width = 0.35
    ax6.bar(x - width/2, avg_accs, width, label='Avg Accuracy', alpha=0.8, color='steelblue')
    ax6_twin = ax6.twinx()
    ax6_twin.bar(x + width/2, avg_mis, width, label='Avg MI (bits)', alpha=0.8, color='coral')
    
    ax6.set_ylabel('Average Accuracy', fontweight='bold', fontsize=10)
    ax6_twin.set_ylabel('Average MI (bits)', fontweight='bold', fontsize=10)
    ax6.set_xlabel('Category', fontweight='bold', fontsize=10)
    ax6.set_title('Average Metrics by Category', fontweight='bold', fontsize=11)
    ax6.set_xticks(x)
    ax6.set_xticklabels(cat_names)
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(alpha=0.3)
    
    # 7. Ranked accuracy
    ax7 = fig.add_subplot(gs[2, 0])
    sorted_idx = np.argsort(accs)[::-1]
    colors_ranked = [{'Weak': '#ff6b6b', 'Semi-Weak': '#ffa500', 'Strong': '#4caf50'}.get(categories[i], 'gray') for i in sorted_idx]
    ax7.barh(range(len(systems)), accs[sorted_idx], color=colors_ranked, alpha=0.8, edgecolor='black')
    ax7.set_yticks(range(len(systems)))
    ax7.set_yticklabels([systems[i] for i in sorted_idx], fontsize=7)
    ax7.set_xlabel('Accuracy', fontweight='bold', fontsize=10)
    ax7.set_title('Systems Ranked by Accuracy', fontweight='bold', fontsize=11)
    ax7.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax7.grid(alpha=0.3, axis='x')
    
    # 8. Ranked MI
    ax8 = fig.add_subplot(gs[2, 1])
    sorted_idx_mi = np.argsort(mi_bits)[::-1]
    colors_ranked_mi = [{'Weak': '#ff6b6b', 'Semi-Weak': '#ffa500', 'Strong': '#4caf50'}.get(categories[i], 'gray') for i in sorted_idx_mi]
    ax8.barh(range(len(systems)), mi_bits[sorted_idx_mi], color=colors_ranked_mi, alpha=0.8, edgecolor='black')
    ax8.set_yticks(range(len(systems)))
    ax8.set_yticklabels([systems[i] for i in sorted_idx_mi], fontsize=7)
    ax8.set_xlabel('MI Estimate (bits)', fontweight='bold', fontsize=10)
    ax8.set_title('Systems Ranked by MI', fontweight='bold', fontsize=11)
    ax8.grid(alpha=0.3, axis='x')
    
    # 9. Summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')
    
    table_data = []
    for cat in ['Weak', 'Semi-Weak', 'Strong']:
        mask = categories == cat
        if mask.any():
            n = mask.sum()
            avg_acc = accs[mask].mean()
            avg_adv = advantages[mask].mean()
            avg_mi = mi_bits[mask].mean()
            success_rate = (accs[mask] > 0.6).sum() / n
            table_data.append([cat, n, f'{avg_acc:.3f}', f'{avg_adv:.3f}', f'{avg_mi:.2f}', f'{success_rate:.1%}'])
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Category', 'N', 'Avg Acc', 'Avg Adv', 'Avg MI', 'Success'],
                     cellLoc='center', loc='center', colWidths=[0.2, 0.1, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    ax9.set_title('Summary Statistics', fontweight='bold', fontsize=11, pad=20)
    
    plt.savefig('indcpa_mine_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: indcpa_mine_analysis.png")
    
    return fig


def plot_mine_convergence(history_dict):
    """
    Plot MINE convergence curves for multiple systems.
    
    Args:
        history_dict: Dict mapping system_name -> mine_history list
    
    Usage:
        history_dict = {}
        for sysname in systems_to_test:
            res = run_experiment(...)
            history_dict[sysname] = res['mine_history']
        
        plot_mine_convergence(history_dict)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    WEAK_SYSTEMS = ['No Encryption', 'Constant XOR', 'Toy Fixed XOR', 'Toy Substitution']
    SEMI_WEAK_SYSTEMS = ['Semi Reduced Feistel', 'Semi Partial Mask', 'AES CTR Reduced']
    STRONG_SYSTEMS = ['AES ECB', 'AES CTR', 'DES']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Weak systems
    ax = axes[0]
    for sys, history in history_dict.items():
        if sys in WEAK_SYSTEMS:
            epochs = range(1, len(history) + 1)
            mi_bits = np.array(history) / np.log(2)
            ax.plot(epochs, mi_bits, label=sys, linewidth=2, alpha=0.7)
    ax.set_ylabel('MI Estimate (bits)', fontweight='bold', fontsize=11)
    ax.set_title('MINE Convergence - Weak Systems', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    # Semi-weak systems
    ax = axes[1]
    for sys, history in history_dict.items():
        if sys in SEMI_WEAK_SYSTEMS:
            epochs = range(1, len(history) + 1)
            mi_bits = np.array(history) / np.log(2)
            ax.plot(epochs, mi_bits, label=sys, linewidth=2, alpha=0.7)
    ax.set_ylabel('MI Estimate (bits)', fontweight='bold', fontsize=11)
    ax.set_title('MINE Convergence - Semi-Weak Systems', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    # Strong systems
    ax = axes[2]
    for sys, history in history_dict.items():
        if sys in STRONG_SYSTEMS:
            epochs = range(1, len(history) + 1)
            mi_bits = np.array(history) / np.log(2)
            ax.plot(epochs, mi_bits, label=sys, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax.set_ylabel('MI Estimate (bits)', fontweight='bold', fontsize=11)
    ax.set_title('MINE Convergence - Strong Systems', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mine_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mine_convergence.png")
    
    return fig


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
    results_summary = []
    history_dict = {}
    
    for sysname in systems_to_test:
        try:
            res =  run_experiment(
                cipher_name=sysname,
                keys=keys,
                indcpa_samples=args.indcpa_samples,
                indcpa_epochs=args.indcpa_epochs,
                mine_samples=args.mine_samples,
                mine_epochs=args.mine_epochs,
                device=args.device
            )

            results_summary.append((sysname, res['adv_accuracy'], res['mi_estimate']))
            history_dict[sysname] = res['mine_history']  # Store for convergence plot
        except Exception as e:
            print(f"[!] Error testing {sysname}: {e}")
    
    # Generate plots
    plot_indcpa_mine_analysis(results_summary)
    plot_mine_convergence(history_dict)