"""
TESS Variable Star Hunter Pipeline
Version: 1.0.0
Author: [Your Name/GitHub Handle]
Description: Automated pipeline to detect and validate Eclipsing Binaries in TESS Sector data.
"""

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SECTOR_TO_SCAN = 55
SEARCH_COORDS = SkyCoord(ra=270.0, dec=60.0, unit=(u.deg, u.deg)) # Draco
SEARCH_RADIUS = 0.5 * u.deg
MAGNITUDE_RANGE = [10, 13]  # Tmag

def fetch_targets():
    """Step 1: Fetch a list of stars from the MAST Catalog."""
    print(f"--- FETCHING TARGETS IN SECTOR {SECTOR_TO_SCAN} REGION ---")
    print(f"Coords: {SEARCH_COORDS.to_string()}")
    
    catalog_data = Catalogs.query_region(
        SEARCH_COORDS,
        radius=SEARCH_RADIUS,
        catalog="TIC"
    )
    
    df = catalog_data.to_pandas()
    # Filter for magnitude
    df = df[(df['Tmag'] >= MAGNITUDE_RANGE[0]) & (df['Tmag'] <= MAGNITUDE_RANGE[1])]
    df = df.sort_values("Tmag").reset_index(drop=True)
    
    target_list = df['ID'].astype(str).tolist()
    print(f"✅ FOUND {len(target_list)} TARGETS.")
    return target_list

def run_bls_analysis(lc, target_id):
    """Step 2: Run Box Least Squares to find periodic signals."""
    # Flatten the light curve to remove long-term trends
    clean_lc = lc.remove_nans().normalize().flatten(window_length=1001)
    
    # Run BLS
    periodogram = clean_lc.to_periodogram(method='bls', period=np.linspace(0.5, 12, 50000), frequency_factor=500)
    
    max_power = periodogram.max_power.value
    best_period = periodogram.period_at_max_power.value
    best_t0 = periodogram.transit_time_at_max_power.value
    
    # Calculate SNR
    median_power = np.median(periodogram.power.value)
    std_power = np.std(periodogram.power.value)
    snr = (max_power - median_power) / std_power
    
    return best_period, best_t0, max_power, snr, clean_lc

def validate_candidate(target_id, sector=None):
    """Step 3: Deep Dive Validation with Alias Check."""
    print(f"\n🔬 STARTING DEEP DIVE: TIC {target_id}")
    
    try:
        search = lk.search_lightcurve(f"TIC {target_id}", mission="TESS", sector=sector)
        if len(search) == 0:
            # Fallback to any sector if specific one not found
            search = lk.search_lightcurve(f"TIC {target_id}", mission="TESS")
            
        if len(search) == 0:
            print("❌ No data found.")
            return

        lc = search[0].download()
        period, t0, power, snr, clean_lc = run_bls_analysis(lc, target_id)
        
        print(f"   > Period: {period:.5f} d")
        print(f"   > Power:  {power:.1f}")
        print(f"   > SNR:    {snr:.2f}")

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Standard Fold
        folded = clean_lc.fold(period=period, epoch_time=t0)
        ax[0].scatter(folded.phase.value, folded.flux.value, s=1, c='k', alpha=0.3)
        ax[0].set_title(f"TIC {target_id}: P = {period:.4f} d")
        ax[0].set_xlabel("Phase")
        ax[0].set_ylabel("Normalized Flux")

        # 2. Alias Check (Double Period)
        double_period = period * 2
        folded_2 = clean_lc.fold(period=double_period, epoch_time=t0)
        ax[1].scatter(folded_2.phase.value, folded_2.flux.value, s=1, c='r', alpha=0.3)
        ax[1].set_title(f"ALIAS CHECK (P x 2): {double_period:.4f} d")
        ax[1].set_xlabel("Phase")
        
        plt.tight_layout()
        plt.show()
        print("✅ Validation Complete. Check plot for Primary/Secondary eclipses.")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

def batch_process():
    """Step 4: The Batch Runner."""
    targets = fetch_targets()
    print("\n--- STARTING BATCH SCAN ---")
    
    for i, tic in enumerate(targets[:100]): # Limit to 100 for demo
        try:
            search = lk.search_lightcurve(f"TIC {tic}", mission="TESS", sector=SECTOR_TO_SCAN)
            if len(search) == 0: continue
            
            lc = search[0].download()
            period, t0, power, snr, clean_lc = run_bls_analysis(lc, tic)
            
            # --- FILTERING ---
            # 1. Ignore weak signals
            if power < 150: continue
            
            # 2. Ignore 7-day satellite glitches
            if 6.8 < period < 7.2:
                continue 
            
            print(f"[{i+1}] TIC {tic} | P: {period:.4f} d | Power: {power:.0f} | SNR: {snr:.1f}")
            print("   🌟 STRONG CANDIDATE DETECTED!")
            
            # Quick look plot for candidate
            clean_lc.fold(period, t0).scatter(label=f"TIC {tic}")
            plt.show()
            
        except Exception as e:
            continue

# --- MAIN MENU ---
if __name__ == "__main__":
    print("=== TESS VARIABLE STAR HUNTER ===")
    print("1. Run Batch Scan (Draco Region)")
    print("2. Validate Known Discovery")
    
    choice = input("Select Mode (1 or 2): ")
    
    if choice == "1":
        batch_process()
    elif choice == "2":
        tid = input("Enter TIC ID (e.g., 233066920): ")
        validate_candidate(tid)
    else:
        print("Invalid selection.")
