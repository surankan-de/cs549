# utils.py
import torch
from torch.utils.data import Dataset
import numpy as np

def bytes_to_tensor(b, out_len=None):
    arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32) / 255.0
    if out_len is not None and len(arr) < out_len:
        # pad with zeros
        pad = np.zeros(out_len - len(arr), dtype=np.float32)
        arr = np.concatenate([arr, pad])
    return torch.from_numpy(arr)

class PTCTDataset(Dataset):
    """
    Plaintext-Ciphertext dataset for MINE / classifier.
    items: list of plaintext bytes, ciphertext bytes or list
    """
    def __init__(self, plaintexts, ciphertexts, labels=None, pt_len=16, ct_len=16):
        self.plaintexts = plaintexts
        self.ciphertexts = ciphertexts
        self.labels = labels
        self.pt_len = pt_len
        self.ct_len = ct_len

    def __len__(self):
        return len(self.plaintexts)

    def __getitem__(self, idx):
        pt = self.plaintexts[idx]
        ct = self.ciphertexts[idx]
        # If ct is concatenated blocks, keep only first block (or derive representation)
        pt_tensor = bytes_to_tensor(pt, self.pt_len)
        ct_tensor = bytes_to_tensor(ct, self.ct_len)
        if self.labels is None:
            return pt_tensor, ct_tensor
        else:
            return pt_tensor, ct_tensor, int(self.labels[idx])
