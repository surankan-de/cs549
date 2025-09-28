# data_gen.py
import os
import numpy as np
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Util import Counter
from Crypto.Random import get_random_bytes

BLOCK_SIZE = 16  # bytes for AES

def pad_pkcs7(b, block=16):
    pad_len = block - (len(b) % block)
    return b + bytes([pad_len])*pad_len

def unpad_pkcs7(b):
    pad_len = b[-1]
    return b[:-pad_len]

def aes_ecb_encrypt_batch(plaintexts, key=None):
    if key is None:
        key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_ECB)
    ct = []
    for p in plaintexts:
        p_pad = pad_pkcs7(p, 16)
        ct.append(cipher.encrypt(p_pad))
    return key, b"".join(ct) if False else ct

def aes_ctr_encrypt_batch(plaintexts, key=None, nonce=None, reduced_counter=False, counter_bits=8):
    if key is None:
        key = get_random_bytes(16)
    cts = []
    for i, p in enumerate(plaintexts):
        # create counter: if reduced_counter True, use small counter size to induce collisions
        if nonce is None:
            nonce = get_random_bytes(8)
        if reduced_counter:
            # small counter width -> collisions
            ctr_val = i & ((1 << counter_bits)-1)
            ctr = Counter.new(64, prefix=nonce, initial_value=ctr_val)
        else:
            ctr = Counter.new(64, prefix=nonce)
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
        cts.append(cipher.encrypt(p))
    return key, cts

def des_ecb_encrypt_batch(plaintexts, key=None):
    if key is None:
        key = get_random_bytes(8)
    cipher = DES.new(key, DES.MODE_ECB)
    ct = [cipher.encrypt(pad_pkcs7(p, 8)) for p in plaintexts]
    return key, ct

def rsa_encrypt_batch(plaintexts, key_size=1024):
    key = RSA.generate(key_size)
    pub = key.publickey()
    cipher = PKCS1_OAEP.new(pub)
    cts = [cipher.encrypt(p) for p in plaintexts]
    return key, cts

# helper to generate random plaintexts or class-conditioned plaintexts
def generate_plaintexts_random(n, length=16):
    return [get_random_bytes(length) for _ in range(n)]

def generate_plaintexts_classes(n, length=16, num_classes=10):
    # create num_classes base templates and add small noise per instance
    templates = [get_random_bytes(length) for _ in range(num_classes)]
    samples = []
    labels = []
    per_class = n // num_classes
    for c in range(num_classes):
        for _ in range(per_class):
            # noise: flip some bytes randomly
            t = bytearray(templates[c])
            idx = np.random.choice(length, size=max(1, length//8), replace=False)
            for i in idx:
                t[i] = (t[i] + np.random.randint(0,256)) & 0xFF
            samples.append(bytes(t))
            labels.append(c)
    # if remainder
    while len(samples) < n:
        c = np.random.randint(0, num_classes)
        samples.append(templates[c])
        labels.append(c)
    return samples, np.array(labels)

if __name__ == "__main__":
    # quick sanity run
    pts = generate_plaintexts_random(10, 16)
    k, cts = aes_ecb_encrypt_batch(pts)
    print("AES-ECB sample ct len:", len(cts[0]))
