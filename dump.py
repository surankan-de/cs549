# Check A: repeat encrypt same plaintext many times
from indcpa import generate_keys, encrypt_variant
from collections import Counter

keys = generate_keys()
msg = b'\x00' * 8
N = 200
cts_det = [encrypt_variant('DES', keys, msg) for _ in range(N)]
cts_nd  = [encrypt_variant('DES NonDet', keys, msg) for _ in range(N)]

# How many unique ciphertexts?
print("DES unique cts for same msg:", len(set(cts_det)), " / ", N)
print("DES NonDet unique cts for same msg:", len(set(cts_nd)), " / ", N)

# Show a few hex samples
print("DES samples:", [c.hex() for c in cts_det[:5]])
print("DES NonDet samples:", [c.hex() for c in cts_nd[:5]])
