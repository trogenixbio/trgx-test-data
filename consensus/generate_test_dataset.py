# generate_test_dataset.py
# Generate test data for consensus implementation of enformer pretrained
# Generate attn and gxi for test input of sequences in fa format

from __future__ import annotations
import math, warnings, pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

# grelu helpers
from grelu.attention import get_attention_scores
from grelu.helpers import convert_input_type
from grelu.interpret.score import get_attributions
from grelu import load_model

def summarize_region(t: torch.Tensor | np.ndarray, s_bp: int, e_bp: int, seq_len: int) -> float:
    """Sum tensor values inside [s_bp, e_bp) (bp coordinates)."""
    if t.ndim not in {1, 2}:
        raise ValueError("tensor must be 1‑D or 2‑D")
    bin_size = seq_len / t.shape[-1]
    s_bin = int(s_bp // bin_size)
    e_bin = int(math.ceil(e_bp / bin_size))
    return float(t[..., s_bin:e_bin].sum())

def run_job(cfg: Dict):
    """Run a single configuration dict, write CSV, and return the output path."""
    fasta   = pathlib.Path(cfg["fasta"])
    model_p = pathlib.Path(cfg["model"])
    out_csv = pathlib.Path(cfg.get("output", "test_output.csv"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_p).to(device).eval()

    rows: List[dict] = []
    for idx, rec in enumerate(SeqIO.parse(fasta, "fasta")):
        if idx >= cfg.get("nseq", 50):
            break
        seq_id  = rec.id
        seq_str = str(rec.seq).upper()

        # grad×input 
        gxi_np = get_attributions(
            model,
            seqs=[seq_str],
            method="inputxgradient",
            device=device,
        )[0]                                   # (4, L)
        gxi    = torch.from_numpy(np.abs(gxi_np).sum(axis=0))  # (L,)
        seq_len = gxi.shape[0]

        # attention
        x_onehot = convert_input_type(seq_str, "one_hot").to(device).unsqueeze(0)
        attn_np  = get_attention_scores(model, x_onehot, block_idx=cfg.get("block_idx", -1))
        attn     = torch.from_numpy(attn_np).mean(dim=0)  # (L, L)

        # summarise window
        centre   = cfg.get("centre", seq_len // 2)
        interval = cfg.get("interval", 2000)
        s_bp, e_bp = centre - interval, centre + interval

        rows.append({
            "seq_id":         seq_id,
            "gradxinput_sum": summarize_region(gxi, s_bp, e_bp, seq_len),
            "attention_sum":  summarize_region(attn, s_bp, e_bp, seq_len),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ {out_csv}: {len(rows)} sequences")
    return out_csv

# Example runs
if __name__ == "__main__":

    examples = [
        # 1) 50 sequences, ±2 kb window around the *middle* (centre = 2 000 // 2 = 1 000)
        {
            "fasta":   "data/sequences.fa",
            "model":   "https://storagepublicmodels01.blob.core.windows.net/models/enformer-pytorch.tar.gz",
            "output":  "outputs/test_output_50seq.csv",
            "nseq":    50,
            "interval": 2000,
            "centre":  2000 // 2,
            "block_idx": -1,   # last transformer layer
        },
        # 2) 20 sequences, ±1 kb window, centre = 1 000 // 2 = 500
        {
            "fasta":   "data/sequences.fa",
            "model":   "https://storagepublicmodels01.blob.core.windows.net/models/enformer-pytorch.tar.gz",
            "output":  "outputs/first_layer_attention.csv",
            "nseq":    20,
            "interval": 1000,
            "centre":  1000 // 2,
            "block_idx": -1,    # last transformer layer
        },
    ]

    for cfg in examples:
        try:
            run_job(cfg)
        except Exception as e:
            print(f"✗ failed {cfg['output']}: {e}")
