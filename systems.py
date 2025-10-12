# systems.py - FIXED VERSION
from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
import hashlib

# ------------------- Toy / Diagnostic Ciphers -------------------
import hashlib
from Crypto.Cipher import AES
from Crypto.Util import Counter  # already imported above; kept for clarity

# ---------------- More weak toy ciphers (for experiments) ----------------
import hashlib
# ---------------- Semi-weak / Mid-strength Toy Ciphers ----------------
import hashlib
from Crypto.Util import Counter
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def semi_reduced_feistel(key: bytes, plaintext: bytes, rounds: int = 3) -> bytes:
    """
    Feistel with 3 rounds (nonlinear via SHA256). Moderate mixing -> semi-weak.
    """
    out = bytearray()
    k = key if key is not None else b"semik"
    for i in range(0, len(plaintext), 8):
        block = plaintext[i:i+8].ljust(8, b'\x00')
        L = block[:4]; R = block[4:]
        for r in range(rounds):
            f = hashlib.sha256(R + k + bytes([r])).digest()[:4]
            L, R = R, bytes([l ^ f_i for l, f_i in zip(L, f)])
        out.extend(L + R)
    return bytes(out[:len(plaintext)])

def semi_partial_mask(key: bytes, plaintext: bytes, keep_bits: int = 6) -> bytes:
    """
    Keep top `keep_bits` of each byte and randomize the lower bits.
    Keeps some structure, adds noise to low bits -> semi-weak.
    """
    if keep_bits < 1 or keep_bits > 7:
        keep_bits = 6
    mask = ((1 << keep_bits) - 1) << (8 - keep_bits)
    rnd = get_random_bytes(len(plaintext))
    return bytes([(b & mask) | (r & (~mask & 0xFF)) for b, r in zip(plaintext, rnd)])

def semi_truncated_block(key: bytes, plaintext: bytes, keep_bytes: int = 4) -> bytes:
    """
    Encrypt with AES-ECB and *truncate* to first keep_bytes of the ciphertext,
    then append deterministic padding (e.g., XOR with key-derived stream).
    Truncation reduces entropy but leaves structured relationship to plaintext.
    """
    akey = (key if key is not None and len(key) >= 16 else hashlib.sha256(key if key else b'k').digest()[:16])
    cipher = AES.new(akey, AES.MODE_ECB)
    ct_full = cipher.encrypt(plaintext[:16].ljust(16, b'\x00'))
    keep = ct_full[:keep_bytes]
    # deterministic tail: XOR plaintext first keep_bytes with key bytes
    tail = bytes([p ^ akey[i % len(akey)] for i, p in enumerate(plaintext[:len(plaintext)])])[keep_bytes:keep_bytes+max(0, len(plaintext)-keep_bytes)]
    return (keep + tail)[:max(keep_bytes, len(plaintext))]

def semi_nonce_mix(key: bytes, plaintext: bytes) -> bytes:
    """
    Use a nonce that mixes random + first 4 bytes of plaintext:
    nonce = random(12) || plaintext[:4]
    This leaks a small portion of the plaintext while keeping partial randomness.
    """
    # Take first 4 bytes of plaintext (or pad if shorter)
    pt_part = (plaintext[:4] + b'\x00' * 4)[:4]
    nonce_rand = get_random_bytes(12)
    nonce = nonce_rand + pt_part  # total 16 bytes (128 bits)
    
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, "big"))
    cipher = AES.new(hashlib.sha256(key if key else b'kn').digest()[:16], AES.MODE_CTR, counter=ctr)
    ciphertext = cipher.encrypt(plaintext)
    
    # Return full nonce (so decryption could be done) + ciphertext
    return nonce + ciphertext

def semi_lfsr_longperiod(key: bytes, plaintext: bytes, taps=(7,5,4,3)) -> bytes:
    """
    LFSR with 16-bit state seeded from key => longer period than toy 8-bit LFSR,
    but still short compared to secure ciphers. Gives repeating patterns at scale.
    """
    seed = (int.from_bytes(hashlib.sha256(key if key else b'lfsr').digest()[:2], 'big') or 1) & 0xFFFF
    state = seed
    out = bytearray()
    for i in range(len(plaintext)):
        # generate one byte by stepping LFSR 8 times
        byte = 0
        for _ in range(8):
            newbit = 0
            for t in taps:
                newbit ^= (state >> t) & 1
            state = ((state << 1) & 0xFFFF) | newbit
            byte = ((byte << 1) | (state & 1)) & 0xFF
        out.append(plaintext[i] ^ byte)
    return bytes(out)

def semi_partial_key_rotation(key: bytes, plaintext: bytes, window: int = 3) -> bytes:
    """
    XOR with a rotating short key window: key bytes used rotate every 'window' bytes.
    If window << message len, repeats cause leakage across blocks.
    """
    if not key:
        key = b'\x55\xAA\x33'
    klen = len(key)
    out = bytearray(len(plaintext))
    for i, b in enumerate(plaintext):
        rot_index = ((i // window) % klen)
        out[i] = b ^ key[rot_index]
    return bytes(out)


def toy_caesar(key: bytes, plaintext: bytes, shift: int = None) -> bytes:
    """
    Simple byte-wise Caesar shift (add a constant mod 256).
    Deterministic and extremely weak.
    """
    if shift is None:
        shift = (key[0] if key else 3) % 256
    return bytes([(b + shift) & 0xFF for b in plaintext])

def toy_repeating_key_xor(key: bytes, plaintext: bytes) -> bytes:
    """
    Repeating-key XOR (Vigenere-like). If key is short, patterns leak.
    """
    if not key:
        key = b'\x42'
    out = bytearray(len(plaintext))
    klen = len(key)
    for i, b in enumerate(plaintext):
        out[i] = b ^ key[i % klen]
    return bytes(out)

def toy_byte_rotate(key: bytes, plaintext: bytes, r: int = None) -> bytes:
    """
    Rotate bits inside each byte by r positions (0-7). Deterministic.
    """
    if r is None:
        r = (key[0] % 7) + 1 if key else 3
    def rot(b):
        return ((b << r) & 0xFF) | (b >> (8 - r))
    return bytes(rot(b) for b in plaintext)

def toy_mask_lowbits(key: bytes, plaintext: bytes, keep_high_bits: int = 4) -> bytes:
    """
    Zero out low (8 - keep_high_bits) bits of every byte; keeps only high nibble by default.
    Very lossy but deterministic: identical plaintexts produce identical ciphertexts on high-nibble.
    """
    mask = ((1 << keep_high_bits) - 1) << (8 - keep_high_bits)
    return bytes([b & mask for b in plaintext])

def toy_lfsr_stream(key: bytes, plaintext: bytes, taps=(0,2,3,5)) -> bytes:
    """
    Very small LFSR-based stream cipher with short period. Use first byte of key as seed.
    Not cryptographically secure; leaks strongly due to short period.
    """
    seed = key[0] if key else 0xA5
    # create an 8-bit state
    state = seed & 0xFF
    out = bytearray()
    for _ in range(len(plaintext)):
        # generate next byte by stepping LFSR 8 times
        byte = 0
        for _ in range(8):
            # tap positions are bit indices; compute new bit
            newbit = 0
            for t in taps:
                newbit ^= (state >> t) & 1
            state = ((state << 1) & 0xFF) | newbit
            byte = ((byte << 1) | (state & 1)) & 0xFF
        out.append(plaintext[len(out)] ^ byte)
    return bytes(out)

def toy_hash_feistel_2rounds(key: bytes, plaintext: bytes) -> bytes:
    """
    2-round Feistel using SHA256 as round function on 8-byte blocks.
    Weak (only 2 rounds) but nonlinear — good for MINE to pick up signal.
    """
    out = bytearray()
    k = key if key is not None else b"hf"
    for i in range(0, len(plaintext), 8):
        block = plaintext[i:i+8].ljust(8, b'\x00')
        L = block[:4]
        R = block[4:]
        # round 1
        f1 = hashlib.sha256(R + k + b'\x01').digest()[:4]
        L1 = bytes([l ^ f for l, f in zip(L, f1)])
        # round 2
        f2 = hashlib.sha256(L1 + k + b'\x02').digest()[:4]
        R1 = bytes([r ^ f for r, f in zip(R, f2)])
        out.extend(L1 + R1)
    return bytes(out[:len(plaintext)])


def toy_fixed_xor(key: bytes, plaintext: bytes) -> bytes:
    """Simple deterministic XOR with a repeating single-byte key derived from key."""
    b = key[0] if key and len(key) > 0 else 0xAA
    return bytes([p ^ b for p in plaintext])

def toy_substitution(key: bytes, plaintext: bytes) -> bytes:
    """
    Deterministic substitution cipher using an S-box derived from the key.
    Same plaintext -> same ciphertext for a given key (weak).
    """
    seed = hashlib.sha256(key if key is not None else b"toy").digest()
    perm = list(range(256))
    j = 0
    for i in range(256):
        j = (j + seed[i % len(seed)]) % 256
        perm[i], perm[j] = perm[j], perm[i]
    sbox = bytes(perm)
    return bytes([sbox[b] for b in plaintext])

def toy_perm(key: bytes, plaintext: bytes) -> bytes:
    """
    Deterministic byte-position permutation. Cycle-shift based on key sum.
    Weak but nontrivial.
    """
    if len(plaintext) == 0:
        return plaintext
    shift = sum(key) % len(plaintext) if key else 1
    return plaintext[shift:] + plaintext[:shift]

def toy_single_feistel(key: bytes, plaintext: bytes) -> bytes:
    """
    Very small Feistel-like toy cipher: one round using SHA256-based F.
    Deterministic and intentionally weak. Operates on 8-byte blocks.
    """
    out = bytearray()
    k = key if key is not None else b"toykey"
    for i in range(0, len(plaintext), 8):
        block = plaintext[i:i+8].ljust(8, b'\x00')
        L = block[:4]
        R = block[4:]
        h = hashlib.sha256(R + k).digest()
        F = h[:4]
        newL = R
        newR = bytes([l ^ f for l, f in zip(L, F)])
        out.extend(newL + newR)
    return bytes(out[:len(plaintext)])

def aes_ctr_fixed_nonce(key: bytes, plaintext: bytes) -> bytes:
    """
    AES CTR using a fixed nonce (i.e. deterministic CTR). Weak (not IND-CPA).
    Use only for diagnostic experiments.
    """
    nonce = hashlib.sha256(key if key is not None else b'ctr').digest()[:16]
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, "big"))
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    return cipher.encrypt(plaintext)


# ------------------- Baselines -------------------

def no_encryption(_, plaintext: bytes) -> bytes:
    """No encryption - plaintext = ciphertext"""
    return plaintext

def one_time_pad(_, plaintext: bytes) -> bytes:
    """Perfect security: completely random pad for each encryption"""
    pad = get_random_bytes(len(plaintext))
    return bytes([p ^ k for p, k in zip(plaintext, pad)])

def constant_key_xor(key: bytes, plaintext: bytes) -> bytes:
    """Insecure: XOR with constant key (repeating key stream)"""
    return bytes([p ^ key[i % len(key)] for i, p in enumerate(plaintext)])

# ------------------- DES -------------------

def des_deterministic(key: bytes, plaintext: bytes) -> bytes:
    """DES ECB mode - deterministic, not IND-CPA secure"""
    cipher = DES.new(key, DES.MODE_ECB)
    pad_len = 8 - (len(plaintext) % 8)
    pt = plaintext + bytes([pad_len]) * pad_len
    return cipher.encrypt(pt)

def des_nondeterministic(key: bytes, plaintext: bytes) -> bytes:
    """DES CBC mode with random IV - IND-CPA secure (for single block)"""
    iv = get_random_bytes(8)
    cipher = DES.new(key, DES.MODE_CBC, iv)
    pad_len = 8 - (len(plaintext) % 8)
    pt = plaintext + bytes([pad_len]) * pad_len
    return iv + cipher.encrypt(pt)

# ------------------- AES -------------------

def aes_ecb(key: bytes, plaintext: bytes) -> bytes:
    """AES ECB mode - deterministic, not IND-CPA secure"""
    cipher = AES.new(key, AES.MODE_ECB)
    pad_len = 16 - (len(plaintext) % 16)
    pt = plaintext + bytes([pad_len]) * pad_len
    return cipher.encrypt(pt)

def aes_ctr(key: bytes, plaintext: bytes) -> bytes:
    """AES CTR mode with random nonce - IND-CPA secure"""
    nonce = get_random_bytes(16)
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, "big"))
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    ct = cipher.encrypt(plaintext)
    return nonce + ct

# Global counter for AES CTR Reduced
_aes_ctr_reduced_counter = 0

def aes_ctr_reduced_counter(key: bytes, plaintext: bytes) -> bytes:
    """
    AES CTR with reduced counter space (resets at 100,000).
    
    VULNERABILITY: After 100k encryptions, counter resets causing nonce reuse!
    Nonce reuse in CTR mode means same keystream is reused:
    - C1 = M1 ⊕ KeyStream(nonce)
    - C2 = M2 ⊕ KeyStream(nonce)  [same nonce!]
    - C1 ⊕ C2 = M1 ⊕ M2  [keystream cancels out]
    
    This makes patterns in plaintexts directly visible in ciphertexts.
    """
    global _aes_ctr_reduced_counter
    
    # Counter wraps at 100,000
    counter_val = _aes_ctr_reduced_counter % 100000
    _aes_ctr_reduced_counter += 1
    
    # Use counter as nonce (causes reuse!)
    nonce = counter_val.to_bytes(8, 'big') + b'\x00' * 8
    
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, "big"))
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    ct = cipher.encrypt(plaintext)
    
    return nonce[:8] + ct

# ------------------- RSA -------------------

def rsa_plain(pubkey, plaintext: bytes) -> bytes:
    """
    Textbook RSA (deterministic, no padding).
    
    VULNERABILITY: Same plaintext always gives same ciphertext.
    C = M^e mod n
    """
    n_bytes = (pubkey.size_in_bits() + 7) // 8
    m_int = int.from_bytes(plaintext, 'big')
    c_int = pow(m_int, pubkey.e, pubkey.n)
    return c_int.to_bytes(n_bytes, 'big')

def rsa_oaep(pubkey, plaintext: bytes) -> bytes:
    """
    RSA with OAEP padding (randomized, IND-CPA secure).
    
    OAEP adds random padding before encryption, so:
    - Same plaintext → different ciphertext each time
    - Provides semantic security
    """
    cipher = PKCS1_OAEP.new(pubkey)
    return cipher.encrypt(plaintext)

# Global counter for RSA OAEP Reused
_rsa_oaep_reused_counter = 0

def rsa_oaep_reused_seed(pubkey, plaintext: bytes) -> bytes:
    """
    RSA-OAEP with deterministic randomness source.
    
    CRITICAL VULNERABILITY: Uses plaintext hash as RNG seed!
    - Same plaintext → same seed → same padding → same ciphertext
    - Completely breaks IND-CPA security
    - Makes OAEP behave like textbook RSA
    
    This is what happens when you misuse cryptographic RNGs!
    """
    global _rsa_oaep_reused_counter
    _rsa_oaep_reused_counter += 1
    
    # Use plaintext hash as seed (WRONG!)
    seed = hashlib.sha256(plaintext).digest()
    
    class DeterministicRandom:
        """Deterministic RNG that produces same output for same seed"""
        def __init__(self, seed):
            self.seed = seed
            self.counter = 0
        
        def __call__(self, n):
            result = b''
            while len(result) < n:
                h = hashlib.sha256(self.seed + self.counter.to_bytes(4, 'big')).digest()
                result += h
                self.counter += 1
            return result[:n]
    
    randfunc = DeterministicRandom(seed)
    cipher = PKCS1_OAEP.new(pubkey, randfunc=randfunc)
    return cipher.encrypt(plaintext)

# ------------------- Key generator -------------------

def generate_keys():
    """Generate all keys needed for experiments"""
    keys = {
        "AES": get_random_bytes(16),     # 128-bit AES key
        "DES": get_random_bytes(8),      # 56-bit DES key (8 bytes with parity)
        "XOR": get_random_bytes(16),     # Key for constant XOR
        "RSA": RSA.generate(2048)        # 2048-bit RSA key pair
    }
    return keys

# ------------------- Encryption dispatcher -------------------

def encrypt_variant(name: str, keys, plaintext: bytes) -> bytes:
    """
    Unified interface for all encryption schemes.
    
    Maps cipher names to their implementations.
    """
    if name == 'No Encryption':
        return no_encryption(None, plaintext)
    elif name == 'One-Time Pad':
        return one_time_pad(None, plaintext)
    elif name == 'Constant XOR':
        return constant_key_xor(keys['XOR'], plaintext)
    # Toy / diagnostic (weaker) ciphers
    elif name == 'Toy Fixed XOR':
        return toy_fixed_xor(keys.get('XOR', b'\xAA'), plaintext)
    elif name == 'Toy Substitution':
        return toy_substitution(keys.get('XOR', b'\xAA'), plaintext)
    elif name == 'Toy Permutation':
        return toy_perm(keys.get('XOR', b'\xAA'), plaintext)
    elif name == 'Toy 1-Round Feistel':
        return toy_single_feistel(keys.get('DES', b'\x00'*8), plaintext)
    elif name == 'AES CTR Fixed Nonce':
        return aes_ctr_fixed_nonce(keys.get('AES', b'\x00'*16), plaintext)
    # Existing real ciphers
    elif name == 'DES':
        return des_deterministic(keys['DES'], plaintext)
    elif name == 'DES NonDet':
        return des_nondeterministic(keys['DES'], plaintext)
    elif name == 'AES ECB':
        return aes_ecb(keys['AES'], plaintext)
    elif name == 'AES CTR':
        return aes_ctr(keys['AES'], plaintext)
    elif name == 'AES CTR Reduced':
        return aes_ctr_reduced_counter(keys['AES'], plaintext)
    elif name == 'RSA Plain':
        return rsa_plain(keys['RSA'].publickey(), plaintext)
    elif name == 'RSA OAEP':
        return rsa_oaep(keys['RSA'].publickey(), plaintext)
    elif name == 'RSA OAEP Reused':
        return rsa_oaep_reused_seed(keys['RSA'].publickey(), plaintext)
    elif name == 'Toy Caesar':
        return toy_caesar(keys.get('XOR', b'\x03'), plaintext)
    elif name == 'Toy Repeating XOR':
        return toy_repeating_key_xor(keys.get('XOR', b'\x42'), plaintext)
    elif name == 'Toy Byte Rotate':
        return toy_byte_rotate(keys.get('XOR', b'\x03'), plaintext)
    elif name == 'Toy Mask HighNibble':
        return toy_mask_lowbits(keys.get('XOR', b'\x00'), plaintext, keep_high_bits=4)
    elif name == 'Toy LFSR Stream':
        return toy_lfsr_stream(keys.get('XOR', b'\xA5'), plaintext)
    elif name == 'Toy 2-Round Feistel':
        return toy_hash_feistel_2rounds(keys.get('DES', b'\x00'*8), plaintext)
    elif name == 'Semi Reduced Feistel':
        return semi_reduced_feistel(keys.get('DES', b'\x00'*8), plaintext, rounds=3)
    elif name == 'Semi Partial Mask':
        return semi_partial_mask(keys.get('XOR', b'\x00'), plaintext, keep_bits=6)
    elif name == 'Semi Truncated AES':
        return semi_truncated_block(keys.get('AES', b'\x00'*16), plaintext, keep_bytes=4)
    elif name == 'Semi Nonce Mix':
        return semi_nonce_mix(keys.get('AES', b'\x00'*16), plaintext)
    elif name == 'Semi LFSR Long':
        return semi_lfsr_longperiod(keys.get('XOR', b'\xA5'), plaintext)
    elif name == 'Semi Key Rotation':
        return semi_partial_key_rotation(keys.get('XOR', b'\x55\xAA\x33'), plaintext, window=3)
    else:
        raise ValueError(f"Unknown system: {name}")
    
def get_all_systems():
    """
    Return a list of all supported encryption system names in encrypt_variant().
    Used by indcpa.py to iterate over all systems automatically.
    """
    return [
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
