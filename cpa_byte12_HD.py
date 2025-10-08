#!/usr/bin/env python3
# cpa_byte12_HD.py
# Hamming-Distance CPA targeting byte 12 (0-indexed) of AES-128 round-10 key.
# Usage: python cpa_byte12_HD.py traces.csv

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== CONFIG ====
BYTE_IDX = 12  # <-- target byte (0-indexed)
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aes_power_traces.csv")
# =================

# AES inverse S-box
INV_SBOX = np.array([
 0x52,0x09,0x6A,0xD5,0x30,0x36,0xA5,0x38,0xBF,0x40,0xA3,0x9E,0x81,0xF3,0xD7,0xFB,
 0x7C,0xE3,0x39,0x82,0x9B,0x2F,0xFF,0x87,0x34,0x8E,0x43,0x44,0xC4,0xDE,0xE9,0xCB,
 0x54,0x7B,0x94,0x32,0xA6,0xC2,0x23,0x3D,0xEE,0x4C,0x95,0x0B,0x42,0xFA,0xC3,0x4E,
 0x08,0x2E,0xA1,0x66,0x28,0xD9,0x24,0xB2,0x76,0x5B,0xA2,0x49,0x6D,0x8B,0xD1,0x25,
 0x72,0xF8,0xF6,0x64,0x86,0x68,0x98,0x16,0xD4,0xA4,0x5C,0xCC,0x5D,0x65,0xB6,0x92,
 0x6C,0x70,0x48,0x50,0xFD,0xED,0xB9,0xDA,0x5E,0x15,0x46,0x57,0xA7,0x8D,0x9D,0x84,
 0x90,0xD8,0xAB,0x00,0x8C,0xBC,0xD3,0x0A,0xF7,0xE4,0x58,0x05,0xB8,0xB3,0x45,0x06,
 0xD0,0x2C,0x1E,0x8F,0xCA,0x3F,0x0F,0x02,0xC1,0xAF,0xBD,0x03,0x01,0x13,0x8A,0x6B,
 0x3A,0x91,0x11,0x41,0x4F,0x67,0xDC,0xEA,0x97,0xF2,0xCF,0xCE,0xF0,0xB4,0xE6,0x73,
 0x96,0xAC,0x74,0x22,0xE7,0xAD,0x35,0x85,0xE2,0xF9,0x37,0xE8,0x1C,0x75,0xDF,0x6E,
 0x47,0xF1,0x1A,0x71,0x1D,0x29,0xC5,0x89,0x6F,0xB7,0x62,0x0E,0xAA,0x18,0xBE,0x1B,
 0xFC,0x56,0x3E,0x4B,0xC6,0xD2,0x79,0x20,0x9A,0xDB,0xC0,0xFE,0x78,0xCD,0x5A,0xF4,
 0x1F,0xDD,0xA8,0x33,0x88,0x07,0xC7,0x31,0xB1,0x12,0x10,0x59,0x27,0x80,0xEC,0x5F,
 0x60,0x51,0x7F,0xA9,0x19,0xB5,0x4A,0x0D,0x2D,0xE5,0x7A,0x9F,0x93,0xC9,0x9C,0xEF,
 0xA0,0xE0,0x3B,0x4D,0xAE,0x2A,0xF5,0xB0,0xC8,0xEB,0xBB,0x3C,0x83,0x53,0x99,0x61,
 0x17,0x2B,0x04,0x7E,0xBA,0x77,0xD6,0x26,0xE1,0x69,0x14,0x63,0x55,0x21,0x0C,0x7D
], dtype=np.uint8)

# precompute HW table
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def clean_hex_to_bytes(hexstr):
    s = str(hexstr).strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    s = s.replace(" ", "")
    return bytes.fromhex(s)

def parse_hex_matrix(hex_list):
    """Convert list/array of 32-hex strings -> (N,16) uint8 array"""
    N = len(hex_list)
    out = np.zeros((N, 16), dtype=np.uint8)
    for i, s in enumerate(hex_list):
        b = clean_hex_to_bytes(s)
        if len(b) != 16:
            raise ValueError(f"Row {i} has ciphertext/plaintext length != 16")
        out[i, :] = np.frombuffer(b, dtype=np.uint8)
    return out

def pearson_corr_cols(traces_centered, vec_centered):
    """Compute correlation of vec_centered (N,) with each column of traces_centered (N,S)"""
    num = np.dot(vec_centered, traces_centered)          # (S,)
    denom = np.sqrt(np.sum(vec_centered**2) * np.sum(traces_centered**2, axis=0))
    denom[denom == 0] = 1e-20
    return num / denom

def main(csv_path):
    print("Loading CSV (no skipped lines). This may take a moment...")
    df = pd.read_csv(csv_path, header=None, engine='python', on_bad_lines='skip', skip_blank_lines=True)
    print(f"Rows read: {len(df)}, columns: {df.shape[1]}")

    plaintext_hex = df.iloc[:, 0].astype(str).values
    ciphertext_hex = df.iloc[:, 1].astype(str).values
    traces = df.iloc[:, 2:].astype(np.float32).values
    N, S = traces.shape
    print(f"Traces shape: {N} x {S}")

    print("Parsing plaintext & ciphertext bytes ...")
    pt_bytes = parse_hex_matrix(plaintext_hex)
    ct_bytes = parse_hex_matrix(ciphertext_hex)
    pt_target = pt_bytes[:, BYTE_IDX].astype(np.uint8)
    ct_target = ct_bytes[:, BYTE_IDX].astype(np.uint8)

    # center traces once
    traces_centered = traces - traces.mean(axis=0)
    traces_norm = np.sqrt(np.sum(traces_centered**2, axis=0))
    traces_norm[traces_norm == 0] = 1e-20

    # candidate old states to try
    old_states = {
        "plaintext": pt_target,
        "ciphertext": ct_target,
        "zero": np.zeros_like(ct_target, dtype=np.uint8),
    }

    best = {"model": None, "guess": None, "corr": -1.0, "sample_idx": None, "corr_vector": None}

    print("Running HD CPA (testing old_state options)...")
    for model_name, old_vals in old_states.items():
        print(f" - Testing old_state = {model_name}")
        per_guess_strength = np.zeros(256, dtype=np.float64)
        per_guess_peakidx = np.zeros(256, dtype=np.int32)

        for k in range(256):
            new_vals = INV_SBOX[(ct_target ^ k).astype(np.uint8)]
            xor_vals = (old_vals ^ new_vals).astype(np.uint8)
            hd_vec = HW[xor_vals].astype(np.float64)          # (N,)
            hd_centered = hd_vec - hd_vec.mean()
            corr_vec = pearson_corr_cols(traces_centered, hd_centered)   # (S,)
            abs_corr = np.abs(corr_vec)
            per_guess_strength[k] = abs_corr.max()
            per_guess_peakidx[k] = int(abs_corr.argmax())

        g = int(np.argmax(per_guess_strength))
        g_corr = float(per_guess_strength[g])
        g_idx = int(per_guess_peakidx[g])
        print(f"   model={model_name}: best_guess=0x{g:02X}, corr={g_corr:.6f} @sample={g_idx}")

        if g_corr > best["corr"]:
            # recompute full corr vector for chosen guess
            new_vals = INV_SBOX[(ct_target ^ g).astype(np.uint8)]
            xor_vals = (old_vals ^ new_vals).astype(np.uint8)
            hd_vec = HW[xor_vals].astype(np.float64)
            hd_centered = hd_vec - hd_vec.mean()
            corr_vector = pearson_corr_cols(traces_centered, hd_centered)

            best.update({
                "model": model_name,
                "guess": g,
                "corr": g_corr,
                "sample_idx": g_idx,
                "corr_vector": corr_vector
            })

    print("\n=== Best overall ===")
    print(f"Model: {best['model']}")
    print(f"Recovered key byte {BYTE_IDX}: 0x{best['guess']:02X}")
    print(f"Max correlation: {best['corr']:.6f} at sample {best['sample_idx']}")

    # Plot per-guess strength for chosen model
    chosen_old = old_states[best['model']]
    per_guess = np.zeros(256, dtype=np.float64)
    for k in range(256):
        new_vals = INV_SBOX[(ct_target ^ k).astype(np.uint8)]
        xor_vals = (chosen_old ^ new_vals).astype(np.uint8)
        hd_vec = HW[xor_vals].astype(np.float64)
        hd_centered = hd_vec - hd_vec.mean()
        per_guess[k] = np.abs(pearson_corr_cols(traces_centered, hd_centered)).max()

    plt.figure(figsize=(10,4))
    plt.plot(per_guess, lw=2)
    plt.title(f"HD-CPA (byte {BYTE_IDX}) - best model: {best['model']}")
    plt.xlabel("Key guess (0..255)")
    plt.ylabel("Max |Pearson correlation|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot correlation vs sample for best guess
    plt.figure(figsize=(12,4))
    plt.plot(best['corr_vector'])
    plt.title(f"Correlation across samples for guess 0x{best['guess']:02X} (model={best['model']})")
    plt.xlabel("Sample index")
    plt.ylabel("Pearson correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print("Provide CSV path as argument.")
        sys.exit(1)
    main(CSV_PATH)
