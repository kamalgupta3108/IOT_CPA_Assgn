# cpa_byte12.py
# Usage: python cpa_byte12.py traces.csv
# Requires: numpy, pandas, matplotlib

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BYTE_IDX = 12  # zero-based index (change if you meant 1-based)

def parse_hex_bytes(hexstr):
    s = hexstr.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    return bytes.fromhex(s)

def hamming_weight(arr_uint8):
    # arr_uint8: numpy uint8 array
    return np.unpackbits(arr_uint8.reshape(-1,1).astype(np.uint8), axis=1).sum(axis=1)

def pearson_corr(X, y):
    y = y.astype(np.float64)
    y_c = y - y.mean()
    X_mean = X.mean(axis=0)
    X_c = X - X_mean
    num = (X_c * y_c[:, None]).sum(axis=0)
    denom = np.sqrt((X_c**2).sum(axis=0) * (y_c**2).sum())
    denom[denom == 0] = 1e-20
    return num / denom

def main(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 3:
        raise ValueError("CSV expected: plaintext,ciphertext,sample1,...")

    cts = df.iloc[:,1].astype(str).values
    traces = df.iloc[:,2:].astype(float).values
    n_traces, n_samples = traces.shape
    print(f"Loaded {n_traces} traces × {n_samples} samples")

    ct_bytes = np.zeros((n_traces,16), dtype=np.uint8)
    for i, ch in enumerate(cts):
        b = parse_hex_bytes(ch)
        if len(b) != 16:
            raise ValueError(f"Row {i}: ciphertext length {len(b)} (expected 16)")
        ct_bytes[i,:] = np.frombuffer(b, dtype=np.uint8)

    # Correlation matrix: 256 x n_samples
    corr = np.zeros((256, n_samples), dtype=np.float32)
    peaks = np.zeros(256, dtype=np.float32)
    peak_idx = np.zeros(256, dtype=np.int32)

    for g in range(256):
        pred = hamming_weight((ct_bytes[:, BYTE_IDX] ^ np.uint8(g)).astype(np.uint8))
        c = pearson_corr(traces, pred)
        corr[g,:] = c
        abs_c = np.abs(c)
        peaks[g] = abs_c.max()
        peak_idx[g] = int(np.argmax(abs_c))

    best = int(np.argmax(peaks))
    second = int(np.argsort(peaks)[-2])
    print(f"Best guess for byte {BYTE_IDX}: 0x{best:02X}  peak_corr={peaks[best]:.4f} at sample {peak_idx[best]}")
    print(f"2nd  guess: 0x{second:02X}  peak_corr={peaks[second]:.4f} at sample {peak_idx[second]}")

    # plot best vs second
    plt.figure(figsize=(10,4))
    plt.plot(corr[best,:], label=f'best 0x{best:02X}')
    plt.plot(corr[second,:], label=f'2nd  0x{second:02X}')
    plt.xlabel('Sample index')
    plt.ylabel('Pearson correlation')
    plt.title(f'CPA correlations for byte {BYTE_IDX}')
    plt.legend()
    out = Path("cpa_byte12_output.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cpa_byte12.py path/to/traces.csv")
        sys.exit(1)
    main(sys.argv[1])
