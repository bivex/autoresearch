import os
import torch
from pathlib import Path
import random

# Constants for HMM
N_TONES = 12
N_TYPES = 12
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_hmm")
DATA_FILE = os.path.join(CACHE_DIR, "data.pt")
MAX_SONGS = 10000  # Limit to 500 songs for fast iterations

def load_ntc_songs(data_dir: Path):
    songs = []
    names = [p.stem for p in sorted(data_dir.glob("*.ntc"))]
    random.shuffle(names) # Shuffle before loading to get a diverse small set
    names = names[:MAX_SONGS]

    for name in names:
        filepath = data_dir / f"{name}.ntc"
        if not filepath.exists(): continue
        song_steps = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if "[" not in line: continue
                notes_str = line[line.find("[") + 1 : line.find("]")]
                notes = [int(n.strip()) for n in notes_str.split(",") if n.strip()]
                vec = torch.zeros(N_TONES)
                for n in notes:
                    vec[n % N_TONES] = 1.0
                song_steps.append(vec)
        if song_steps:
            songs.append(torch.stack(song_steps))
    return songs

def prepare():
    os.makedirs(CACHE_DIR, exist_ok=True)
    synth_dir = Path("tymoczko_code/Code/First step/synth_data")
    if not synth_dir.exists():
        synth_dir = Path("/Volumes/External/Code/Melodica/tymoczko_code/Code/First step/synth_data")
        
    print(f"Loading up to {MAX_SONGS} songs from {synth_dir}...")
    songs = load_ntc_songs(synth_dir)
    
    if not songs:
        print("Error: No songs found!")
        return

    random.shuffle(songs)
    split = int(0.9 * len(songs))
    train_songs = songs[:split]
    val_songs = songs[split:]
    
    torch.save({
        "train": train_songs,
        "val": val_songs,
        "n_types": N_TYPES,
        "n_tones": N_TONES
    }, DATA_FILE)
    
    print(f"Done. Saved {len(train_songs)} train and {len(val_songs)} val songs to {DATA_FILE}")

if __name__ == "__main__":
    prepare()
