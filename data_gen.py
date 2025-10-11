# data_gen.py
import os
import numpy as np
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Util import Counter
from Crypto.Random import get_random_bytes
import secrets

BLOCK_SIZE = 64  # bytes for AES

def xor_encrypt_batch_multikey(plaintexts, labels, num_keys=2):
    """
    Encrypt plaintexts with XOR using a set of keys (one per class/label).
    Labels correspond to key IDs.
    """
    keys = [secrets.token_bytes(len(plaintexts[0])) for _ in range(num_keys)]
    cts = []
    for p, lab in zip(plaintexts, labels):
        k = keys[int(lab)]
        cts.append(bytes([p[i] ^ k[i % len(k)] for i in range(len(p))]))
    return keys, cts
def vigenere_autokey_encrypt_batch(plaintexts, labels, key_len=8, num_classes=4):
    """
    Class-aware Autokey Vigenère-style encryption.
    
    - Each class has its own distinct initial key (secrets.token_bytes).
    - The key is extended using the plaintext itself (autokey).
    """
    keys = [secrets.token_bytes(len(plaintexts[0])//2) for _ in range(num_classes)]
    cts = []

    for p, lab in zip(plaintexts, labels):
        k = keys[int(lab)]
        ct_bytes = bytearray(len(p))
        key_stream = bytearray(k)

        for i in range(len(p)):
            ct_bytes[i] = (p[i] + key_stream[i]) % 256
            key_stream.append(p[i])  # Autokey: extend stream with plaintext

        cts.append(bytes(ct_bytes))

    return keys, cts

def vigenere_like_multikey_encrypt_batch(plaintexts, labels, num_keys=4):
    """
    Vigenere-style encryption with distinct keys per class.
    Labels determine which key to use for each plaintext.
    """
    key_len = len(plaintexts[0])  # Length of each Vigenere key
    # Generate distinct keys for each class
    keys = [secrets.token_bytes(key_len) for _ in range(num_keys)]
    
    cts = []
    for p, lab in zip(plaintexts, labels):
        k = keys[int(lab)]
        ct = bytes([(p[i] + k[i % len(k)]) % 256 for i in range(len(p))])
        cts.append(ct)
    
    return keys, cts

def shift_encrypt_batch_multikey(plaintexts, labels, num_keys=2):
    """
    Encrypt plaintexts with XOR using a set of keys (one per class/label).
    Labels correspond to key IDs.
    """
    keys = []
    used = set()
    while len(keys) < num_keys:
        s = secrets.randbelow(256)
        if s not in used:
            keys.append(s)
            used.add(s)
    cts = []
    for p, lab in zip(plaintexts, labels):
        shift = keys[int(lab)]
        cts.append(bytes([(b + shift) % 256 for b in p]))
    return keys, cts
def rotate_encrypt_batch_multikey(plaintexts, labels, num_keys=2):
    """
    Encrypt plaintexts using class-dependent distinct rotations.
    Each label corresponds to a unique random rotation.
    """
    # Generate distinct random rotations for each class
    rotations = []
    used = set()
    while len(rotations) < num_keys:
        r = secrets.randbelow(16)  # rotation < plaintext length
        if r not in used:
            rotations.append(r)
            used.add(r)
    
    cts = []
    for p, lab in zip(plaintexts, labels):
        rotation_actual = rotations[int(lab)] % len(p)
        rotated = p[rotation_actual:] + p[:rotation_actual]
        cts.append(rotated)
    
    return rotations, cts
def affine_encrypt_batch_multikey(plaintexts, labels, num_keys=2):
    """
    Encrypt plaintexts using class-dependent distinct affine transformations.
    Affine cipher: E(x) = (a * x + b) % 256
    Each label corresponds to a unique (a, b) pair.
    """
    # Generate distinct 'a' values (coprime with 256) and random 'b' values
    def coprime_with_256(x):
        from math import gcd
        return gcd(x, 256) == 1

    a_values = []
    used_a = set()
    while len(a_values) < num_keys:
        a = secrets.randbelow(255) + 1  # 1..255
        if a not in used_a and coprime_with_256(a):
            a_values.append(a)
            used_a.add(a)

    b_values = [secrets.randbelow(256) for _ in range(num_keys)]

    # Encrypt
    cts = []
    for p, lab in zip(plaintexts, labels):
        a = a_values[int(lab)]
        b = b_values[int(lab)]
        ct = bytes([(a * byte + b) % 256 for byte in p])
        cts.append(ct)

    keys = list(zip(a_values, b_values))
    return keys, cts

def caesar_encrypt_batch(plaintexts, shift=3):
    cts = [bytes([(b + shift) % 256 for b in p]) for p in plaintexts]
    return shift, cts

def substitution_encrypt_batch(plaintexts, labels, num_classes=4, sboxes=None):
    """Class-aware substitution cipher (one S-box per label/class)"""
    if sboxes is None:
        sboxes = []
        for _ in range(num_classes):
            sbox = np.arange(256, dtype=np.uint8)
            np.random.shuffle(sbox)
            sboxes.append(sbox)
    cts = []
    for p, lab in zip(plaintexts, labels):
        sbox = sboxes[int(lab)]
        cts.append(bytes([sbox[b] for b in p]))
    return sboxes, cts



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
    return key, ct

def aes_ctr_encrypt_batch(plaintexts, key=None, reduced_counter=False, counter_bits=8):
    if key is None:
        key = get_random_bytes(16)
    cts = []
    for i, p in enumerate(plaintexts):
        nonce = get_random_bytes(8)  # NEW nonce per plaintext
        if reduced_counter:
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
    """Generate truly random plaintexts using cryptographically secure randomness"""
    return [secrets.token_bytes(length) for _ in range(n)]

def generate_plaintexts_classes(n, length=16, num_classes=10):
    """
    Generate truly random plaintexts with random class labels.
    This removes any correlation between plaintext content and labels,
    preventing the model from learning plaintext patterns instead of crypto weaknesses.
    """
    # Generate completely random plaintexts
    samples = [secrets.token_bytes(length) for _ in range(n)]
    
    # Assign random labels (no correlation with plaintext)
    labels = np.random.randint(0, num_classes, size=n)
    
    return samples, labels

def generate_plaintexts_structured(n, length=16):
    """Generate plaintexts with some structure (for specific cryptanalytic attacks)"""
    plaintexts = []
    for i in range(n):
        # Create structured data: counter + random bytes
        counter = i.to_bytes(4, 'big')
        random_part = secrets.token_bytes(length - 4)
        plaintexts.append(counter + random_part)
    return plaintexts

def generate_plaintexts_repeated_blocks(n, block_size=16):
    """Generate plaintexts with repeated blocks (useful for ECB mode analysis)"""
    plaintexts = []
    for _ in range(n):
        # Create a plaintext with repeated blocks
        block = secrets.token_bytes(block_size)
        num_repeats = np.random.randint(2, 5)  # 2-4 repeated blocks
        plaintexts.append(block * num_repeats)
    return plaintexts

def generate_plaintexts_binary_labels(n, length=16):
    """Generate random plaintexts with binary labels for distinguisher attacks"""
    plaintexts = generate_plaintexts_random(n, length)
    labels = np.random.randint(0, 2, size=n)
    return plaintexts, labels

def generate_distinguisher_dataset(n_samples, algorithm='aes_ecb', plaintext_length=16):
    """
    Generate dataset for distinguishing encrypted vs random data
    Returns: (data, labels) where label 0 = random, label 1 = encrypted
    """
    half_samples = n_samples // 2
    
    # Generate truly random plaintexts
    plaintexts = generate_plaintexts_random(half_samples, plaintext_length)
    
    # Encrypt plaintexts
    if algorithm == 'aes_ecb':
        _, ciphertexts = aes_ecb_encrypt_batch(plaintexts)
    elif algorithm == 'aes_ctr':
        _, ciphertexts = aes_ctr_encrypt_batch(plaintexts)
    elif algorithm == 'des_ecb':
        _, ciphertexts = des_ecb_encrypt_batch(plaintexts)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Generate random data (same length as ciphertexts)
    random_data = []
    for ct in ciphertexts:
        random_data.append(secrets.token_bytes(len(ct)))
    
    # Combine data
    all_data = ciphertexts + random_data
    labels = np.array([1] * len(ciphertexts) + [0] * len(random_data))
    
    # Shuffle
    indices = np.random.permutation(len(all_data))
    shuffled_data = [all_data[i] for i in indices]
    shuffled_labels = labels[indices]
    
    return shuffled_data, shuffled_labels

def generate_key_recovery_dataset(n_samples, num_keys=2, algorithm='aes_ecb', plaintext_length=16):
    """
    Generate dataset for key recovery attacks
    Returns: (ciphertexts, key_labels)
    """
    samples_per_key = n_samples // num_keys
    all_ciphertexts = []
    all_labels = []
    
    for key_id in range(num_keys):
        # Generate random plaintexts
        plaintexts = generate_plaintexts_random(samples_per_key, plaintext_length)
        
        # Encrypt with same key
        if algorithm == 'aes_ecb':
            key, ciphertexts = aes_ecb_encrypt_batch(plaintexts)
        elif algorithm == 'aes_ctr':
            key, ciphertexts = aes_ctr_encrypt_batch(plaintexts)
        elif algorithm == 'des_ecb':
            key, ciphertexts = des_ecb_encrypt_batch(plaintexts)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        all_ciphertexts.extend(ciphertexts)
        all_labels.extend([key_id] * len(ciphertexts))
    
    # Shuffle
    labels = np.array(all_labels)
    indices = np.random.permutation(len(all_ciphertexts))
    shuffled_ciphertexts = [all_ciphertexts[i] for i in indices]
    shuffled_labels = labels[indices]
    
    return shuffled_ciphertexts, shuffled_labels

def convert_to_numpy_array(byte_list):
    """Convert list of bytes to numpy array for ML models"""
    max_len = max(len(b) for b in byte_list)
    array = np.zeros((len(byte_list), max_len), dtype=np.uint8)
    for i, b in enumerate(byte_list):
        array[i, :len(b)] = list(b)
    return array

if __name__ == "__main__":
    print("Testing improved data generation...")
    
    # Test random plaintext generation
    pts_random = generate_plaintexts_random(10, 16)
    print(f"Random plaintexts generated: {len(pts_random)}")
    print(f"Sample: {pts_random[0].hex()}")
    
    # Test encryption
    k, cts = aes_ecb_encrypt_batch(pts_random)
    print(f"AES-ECB sample ct len: {len(cts[0])}")
    
    # Test class generation (now truly random)
    pts_class, labels = generate_plaintexts_classes(100, 16, 5)
    print(f"Generated {len(pts_class)} plaintexts with {len(np.unique(labels))} classes")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Test distinguisher dataset
    data, labels = generate_distinguisher_dataset(1000, 'aes_ecb', 16)
    print(f"Distinguisher dataset: {len(data)} samples, label distribution: {np.bincount(labels)}")
    
    print("All tests passed!")