# Deep Sky Surveyor (TESS Variable Star Hunter)
# Automated systematic scanner for TESS Full Frame Images.
# Features: Resume capability, Artifact Filtering, Network Retry.

import os
import sys
import time
import json
import warnings
import shutil
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION ---
TARGET_SECTORS = None    # Set to [15, 16] to limit sectors, or None for 'All Available'
SEARCH_RADIUS = 0.3      # Size of each tile (degrees)
MAG_RANGE = (9, 13.0)    # TESS Magnitude range
BATCH_TIME_LIMIT = 3600  # Save and zip every 3600 seconds (1 hour)
MAX_RETRIES = 3          # Network retry attempts

# Artifact Filtering Constants
MIN_DEPTH = 0.005        # 0.5% minimum depth
ALIAS_TOL = 0.01         # Tolerance for integer period artifacts

# Paths
WORK_DIR = "Survey_Data"
CHECKPOINT_FILE = os.path.join(WORK_DIR, "checkpoint.json")
RESULTS_CSV = os.path.join(WORK_DIR, "Candidates_Log.csv")

# --- LIBRARIES ---
# Late imports to allow setup checks
try:
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares
    from astropy import units as u
    from astroquery.mast import Catalogs
except ImportError:
    print("❌ Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

warnings.simplefilter('ignore')

# --- SETUP ENV ---
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

def is_colab():
    return 'google.colab' in sys.modules

# --- STATE MANAGEMENT ---
def load_state():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    else:
        # Generate Grid (Global Sky)
        ra_steps = np.arange(0, 360, 2)    # Every 2 deg RA
        dec_steps = np.arange(-80, 80, 2)  # Every 2 deg DEC
        tiles = [f"{r}_{d}" for r in ra_steps for d in dec_steps]
        
        return {
            "tiles_total": len(tiles),
            "current_tile_index": 0,
            "candidates_found": 0,
            "start_time": time.time(),
            "tiles_list": tiles
        }

def save_state(state):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f)

# --- ANALYSIS ENGINE ---
def is_artifact(period, depth):
    """Returns True if signal is likely noise/alias."""
    if depth < MIN_DEPTH: return True, "Shallow"
    if abs(period - round(period)) < ALIAS_TOL: return True, "Integer Artifact"
    if abs((period % 1) - 0.5) < ALIAS_TOL: return True, "0.5d Alias"
    if period < 0.6 or period > 14.0: return True, "Grid Edge"
    return False, "Clean"

def analyze_star(tic_id):
    try:
        # 1. Search (With Retry)
        search = None
        for _ in range(MAX_RETRIES):
            try:
                search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=TARGET_SECTORS)
                break
            except: time.sleep(2)
        
        if not search or len(search) == 0: return "No Data"

        # 2. Download
        lc = None
        for _ in range(MAX_RETRIES):
            try:
                lc = search[0].download(quality_bitmask='default')
                break
            except: time.sleep(2)
            
        if lc is None: return "Download Fail"
        if np.nanmedian(lc.flux.value) <= 0: return "Bad Flux"

        # 3. Clean & Bin
        lc = lc.normalize().remove_nans().remove_outliers(sigma=4)
        lc_binned = lc.bin(time_bin_size=30*u.min)
        t, y, dy = lc_binned.time.value, lc_binned.flux.value, lc_binned.flux_err.value

        # 4. BLS Search
        period_grid = np.linspace(0.6, 14, 40000)
        durations = np.linspace(0.04, 0.15, 10)
        
        model = BoxLeastSquares(t, y, dy=dy)
        pg = model.power(period_grid, durations)
        
        best_idx = np.argmax(pg.power)
        best_p = pg.period[best_idx]
        depth = pg.depth[best_idx]

        # 5. Filter
        is_bad, reason = is_artifact(best_p, depth)
        if is_bad: return f"Filtered: {reason}"

        # 6. Save Candidate
        best_t0 = pg.transit_time[best_idx]
        epoch = best_t0 + 2457000.0
        
        # Generate Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(t, y, s=1, c='k', alpha=0.3)
        ax[0].set_title(f"TIC {tic_id}")
        
        folded = lc.fold(period=best_p, epoch_time=best_t0)
        folded.scatter(ax=ax[1], s=1, c='purple', alpha=0.5)
        ax[1].set_title(f"P={best_p:.4f} d | D={depth:.1%}")
        
        plot_path = os.path.join(WORK_DIR, f"TIC_{tic_id}_P{best_p:.2f}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return {
            "TIC": tic_id, "RA": lc.ra, "DEC": lc.dec,
            "Period": best_p, "Epoch": epoch, "Depth": depth
        }

    except Exception as e:
        return "Error"

# --- MAIN EXECUTION ---
def main():
    print("🔭 Deep Sky Surveyor Initialized.")
    state = load_state()
    
    # Init CSV
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w") as f: 
            f.write("TIC,RA,DEC,Period,Epoch,Depth\n")

    tiles = state['tiles_list']
    start_time = time.time()
    batch_start = time.time()

    print(f"📍 Resuming at Tile {state['current_tile_index']}/{state['tiles_total']}")
    print(f"💎 Candidates found so far: {state['candidates_found']}")

    # --- GRID LOOP ---
    for i in range(state['current_tile_index'], len(tiles)):
        
        # Hourly Batch Save
        if time.time() - batch_start > BATCH_TIME_LIMIT:
            print(f"\n⏰ Hourly Save Point Reached. Archiving...")
            shutil.make_archive("Survey_Backup", 'zip', WORK_DIR)
            if is_colab():
                from google.colab import files
                files.download('Survey_Backup.zip')
            batch_start = time.time()

        # Parse Tile
        r_str, d_str = tiles[i].split("_")
        ra, dec = float(r_str), float(d_str)

        try:
            # Query MAST
            cat = Catalogs.query_region(f"{ra} {dec}", radius=SEARCH_RADIUS, catalog="TIC")
            df = cat.to_pandas()
            df = df[ (df['Tmag'] >= MAG_RANGE[0]) & (df['Tmag'] <= MAG_RANGE[1]) ]

            print(f"\n📡 Scanning Tile RA:{ra} DEC:{dec} ({len(df)} stars)...")

            for j, row in df.iterrows():
                tic = str(row['ID'])
                
                # Polite Rate Limit
                time.sleep(0.1)

                res = analyze_star(tic)

                if isinstance(res, dict):
                    print(f"   💎 FOUND! TIC {tic} (P={res['Period']:.4f}d)")
                    with open(RESULTS_CSV, "a") as f:
                        f.write(f"{res['TIC']},{res['RA']},{res['DEC']},{res['Period']},{res['Epoch']},{res['Depth']}\n")
                    state['candidates_found'] += 1
                else:
                    # Overwrite line for cleanliness
                    sys.stdout.write(f"\r   [{j+1}/{len(df)}] TIC {tic}: {res}")
                    sys.stdout.flush()

        except Exception as e:
            print(f"\n⚠️ Tile Error: {e}")

        # Update State
        state['current_tile_index'] = i + 1
        save_state(state)

    print("\n🏁 Survey Complete.")

if __name__ == "__main__":
    main()
