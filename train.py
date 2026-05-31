import torch
import numpy as np
import time
import os
import math
import sys

# ---------------------------------------------------------------------------
# Training Constants & Hyperparameters (Agent will modify these)
# ---------------------------------------------------------------------------

ON_PROB = 0.90           # Initial emission prob for chord tones
PRIOR_STRENGTH = 0.01    # Dirichlet prior for transitions
MAX_SELF_LOOP = 0.40     # Cap for chord self-repetitions
EPS = 1e-8
TIME_BUDGET = 300        # 5 minutes fixed budget

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_hmm")
DATA_FILE = os.path.join(CACHE_DIR, "data.pt")
BEST_BPB_FILE = "overall_best_bpb.txt"
BEST_WEIGHTS_FILE = "best_weights.pt"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    try:
        print(f"Starting HMM training on {DEVICE}...", flush=True)
        if not os.path.exists(DATA_FILE):
            print("Error: Data file not found. Run prepare.py first.", flush=True)
            return

        data = torch.load(DATA_FILE, map_location=DEVICE)
        train_songs = data["train"]
        val_songs = data["val"]
        N_TYPES = data["n_types"]
        N_TONES = data["n_tones"]
        print(f"Loaded {len(train_songs)} train and {len(val_songs)} val songs.", flush=True)

        # Index helpers
        r_next_indices = torch.arange(N_TONES, device=DEVICE).view(N_TONES, 1)
        interval_indices = torch.arange(N_TONES, device=DEVICE).view(1, N_TONES)
        r_prev_indices = (r_next_indices - interval_indices) % N_TONES
        r_next_indices_for_beta = (r_next_indices + interval_indices) % N_TONES

        # Initialize HMM parameters
        CHORD_NOTES = {
            0: {0, 4, 7}, 1: {0, 3, 7}, 2: {0, 3, 6}, 3: {0, 4, 8},
            4: {0, 2, 7}, 5: {0, 5, 7}, 6: {0, 4, 7, 11}, 7: {0, 3, 7, 10},
            8: {0, 4, 7, 10}, 9: {0, 4, 7, 11, 2}, 10: {0, 3, 7, 10, 2}, 11: {0, 4, 7, 2},
        }

        pnote = torch.full((N_TONES, N_TYPES), 0.12, device=DEVICE)
        for t_idx, notes in CHORD_NOTES.items():
            for n in notes:
                pnote[n, t_idx] = ON_PROB

        pchord = torch.ones(N_TYPES, device=DEVICE) / N_TYPES
        pchange = torch.ones(N_TYPES, N_TONES, N_TYPES, device=DEVICE) / (N_TONES * N_TYPES)
        for t in range(N_TYPES):
            pchange[t, 0, t] = 2.0
        pchange /= pchange.sum(dim=(1, 2), keepdim=True)

        # ---------------------------------------------------------------------------
        # Training Loop
        # ---------------------------------------------------------------------------

        t_start = time.time()
        iter_idx = 0
        
        print("Beginning EM cycles...", flush=True)
        while time.time() - t_start < TIME_BUDGET:
            total_train_ll = 0.0
            note_hist = torch.zeros_like(pnote)
            chord_hist = torch.zeros(N_TYPES, device=DEVICE)
            change_hist = torch.zeros_like(pchange)

            log_pnote = torch.log(pnote + EPS)
            log_not_pnote = torch.log(1.0 - pnote + EPS)
            diff = log_pnote - log_not_pnote
            emission_bias = log_not_pnote.sum(dim=0)

            for s_idx, song in enumerate(train_songs):
                T = song.shape[0]
                
                # 1. Emission probs
                psets_log = torch.zeros(T, N_TONES, N_TYPES, device=DEVICE)
                for r in range(N_TONES):
                    heard = torch.roll(song, shifts=-r, dims=1)
                    psets_log[:, r, :] = torch.einsum("tp,pk->tk", heard, diff) + emission_bias
                psets = torch.exp(psets_log)
                
                # 2. Forward
                alpha = torch.zeros(T, N_TONES, N_TYPES, device=DEVICE)
                alpha[0] = (pchord / N_TONES) * psets[0]
                norm = alpha[0].sum()
                alpha[0] /= (norm + EPS)
                total_train_ll += torch.log(norm + EPS)

                for t in range(1, T):
                    prev_expanded = alpha[t-1][r_prev_indices]
                    combined_prev = torch.einsum("rik,kio->ro", prev_expanded, pchange)
                    alpha[t] = combined_prev * psets[t]
                    norm = alpha[t].sum()
                    alpha[t] /= (norm + EPS)
                    total_train_ll += torch.log(norm + EPS)

                # 3. Backward
                beta = torch.zeros(T, N_TONES, N_TYPES, device=DEVICE)
                beta[-1] = 1.0
                for t in range(T-2, -1, -1):
                    next_val = psets[t+1] * beta[t+1]
                    next_expanded = next_val[r_next_indices_for_beta]
                    beta[t] = torch.einsum("rio,kio->rk", next_expanded, pchange)
                    beta[t] /= (beta[t].sum() + EPS)

                # 4. Stats
                gamma = alpha * beta
                gamma /= (gamma.sum(dim=(1, 2), keepdim=True) + EPS)
                chord_hist += gamma.sum(dim=(0, 1))
                
                for r in range(N_TONES):
                    heard = torch.roll(song, shifts=-r, dims=1)
                    note_hist += heard.T @ gamma[:, r, :]

                term_next = psets[1:] * beta[1:]
                term_prev_expanded = alpha[:-1][:, r_prev_indices]
                change_hist += torch.einsum("trik,kio,tro->kio", term_prev_expanded, pchange, term_next)

            # 5. M-Step
            pnote = note_hist / (chord_hist.view(1, N_TYPES) + EPS)
            pnote = torch.clamp(pnote, 0.001, 0.999)
            
            uniform_prior = torch.ones_like(change_hist) / (N_TONES * N_TYPES)
            pchange = (change_hist + PRIOR_STRENGTH * uniform_prior) / \
                      (change_hist + PRIOR_STRENGTH * uniform_prior).sum(dim=(1, 2), keepdim=True)

            for t in range(N_TYPES):
                if pchange[t, 0, t] > MAX_SELF_LOOP:
                    excess = pchange[t, 0, t] - MAX_SELF_LOOP
                    pchange[t, 0, t] = MAX_SELF_LOOP
                    others = pchange[t].sum() - pchange[t, 0, t]
                    if others > EPS:
                        pchange[t] *= (1.0 + excess / others)
                        pchange[t, 0, t] = MAX_SELF_LOOP
            pchange /= pchange.sum(dim=(1, 2), keepdim=True)

            iter_idx += 1
            elapsed = time.time() - t_start
            print(f"Iter {iter_idx:3d} | Train LL: {total_train_ll.item():.1f} | Time: {elapsed:.1f}s", flush=True)

        # Evaluation
        print("Evaluating on validation set...", flush=True)
        total_val_ll = 0.0
        total_notes = 0
        with torch.no_grad():
            log_pnote = torch.log(pnote + EPS)
            log_not_pnote = torch.log(1.0 - pnote + EPS)
            diff = log_pnote - log_not_pnote
            emission_bias = log_not_pnote.sum(dim=0)
            for song in val_songs:
                T = song.shape[0]
                total_notes += T * N_TONES
                psets_log = torch.zeros(T, N_TONES, N_TYPES, device=DEVICE)
                for r in range(N_TONES):
                    heard = torch.roll(song, shifts=-r, dims=1)
                    psets_log[:, r, :] = torch.einsum("tp,pk->tk", heard, diff) + emission_bias
                psets = torch.exp(psets_log)
                alpha = torch.zeros(T, N_TONES, N_TYPES, device=DEVICE)
                alpha[0] = (pchord / N_TONES) * psets[0]
                norm = alpha[0].sum()
                alpha[0] /= (norm + EPS)
                total_val_ll += torch.log(norm + EPS)
                for t in range(1, T):
                    prev_expanded = alpha[t-1][r_prev_indices]
                    combined_prev = torch.einsum("rik,kio->ro", prev_expanded, pchange)
                    alpha[t] = combined_prev * psets[t]
                    norm = alpha[t].sum()
                    alpha[t] /= (norm + EPS)
                    total_val_ll += torch.log(norm + EPS)

        val_bpb = -total_val_ll.item() / (math.log(2) * total_notes)
        print(f"\nFinal Validation LL: {total_val_ll.item():.1f}", flush=True)
        print(f"Final val_bpb: {val_bpb:.6f}", flush=True)

        # ---------------------------------------------------------------------------
        # Persistent Saving of Best Weights
        # ---------------------------------------------------------------------------
        overall_best = float('inf')
        if os.path.exists(BEST_BPB_FILE):
            try:
                with open(BEST_BPB_FILE, "r") as f:
                    overall_best = float(f.read().strip())
            except: pass

        if val_bpb < overall_best:
            print(f"New overall best val_bpb: {val_bpb:.6f} (was {overall_best:.6f}). Saving weights...", flush=True)
            with open(BEST_BPB_FILE, "w") as f:
                f.write(f"{val_bpb:.6f}")
            torch.save({
                "pnote": pnote.cpu(),
                "pchange": pchange.cpu(),
                "pchord": pchord.cpu(),
                "val_bpb": val_bpb
            }, BEST_WEIGHTS_FILE)

    except BrokenPipeError:
        sys.stderr.close()

if __name__ == "__main__":
    main()
