# ==============================================================================
# 📦 STEP 0: AUTO-INSTALL DEPENDENCIES
# ==============================================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import lightkurve as lk
except ImportError:
    print("⏳ Installing Lightkurve... (This takes 30 seconds)")
    install("lightkurve")
    import lightkurve as lk
    print("✅ Lightkurve installed.")

# ==============================================================================
# 🚜 STEP 1: CONFIGURATION (HIGH SENSITIVITY MODE)
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import gc
import time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
import warnings

warnings.filterwarnings("ignore")

# ⚙️ SCAN SETTINGS
SECTOR = 75              # Northern CVZ (Best Data)
AUTHOR = "QLP"           # Faint Star Pipeline

# 🎯 TARGET: North Ecliptic Pole (NEP)
CENTER_RA = 270.0
CENTER_DEC = 66.5

# 📏 GRID: 10x10 degrees (100 patches)
GRID_SIZE = 10  
STEP_SIZE = 1.0 

STARS_PER_PATCH = 500  
OUTPUT_FOLDER = "candidates_sensitive"
RESULTS_FILE = "candidates_sensitive.csv"

# ⚡ FILTERS: SENSITIVE MODE
# Lowered thresholds to detect Earths/Neptunes and weak signals.
MIN_SNR = 8.0            # Standard NASA threshold (was 30.0)
MIN_MAG_DEPTH = 0.005    # 0.5% dip (was 5.0%)
MAX_DEPTH_FLUX = 0.90    # Exclude obvious Eclipsing Binaries (>90% blocked)
SUSPICIOUS_PERIODS = [0.5, 1.0, 13.7, 27.4] 
PERIOD_TOLERANCE = 0.05 

# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================
def load_forbidden_catalog(url):
    try:
        r = requests.get(url)
        return {line.split()[0] for line in r.text.split('\n') if len(line.split()) > 0 and line.split()[0].isdigit()}
    except: return set()

def check_vsx(ra, dec):
    try:
        v = Vizier(columns=['OID', 'Name', 'Type'], row_limit=1)
        res = v.query_region(SkyCoord(ra, dec, unit=(u.deg, u.deg)), radius=5*u.arcsec, catalog='B/vsx')
        if len(res) > 0: return True, res[0]['Name']
        return False, None
    except: return False, None

def is_suspicious_period(period):
    for p in SUSPICIOUS_PERIODS:
        if abs(period - p) < (p * PERIOD_TOLERANCE): return True
    return False

def log_candidate(tic, snr, depth, period, ra, dec, status):
    print(f"\n      🌟 HIT FOUND! TIC {tic} | Depth: {depth:.3f} | SNR: {snr:.1f}")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{tic},{snr:.2f},{depth:.4f},{period:.4f},{ra},{dec},{status}\n")

# ==============================================================================
# 🚀 MAIN RASTER SCAN
# ==============================================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("TIC_ID,SNR,Mag_Depth,Period,RA,Dec,Status\n")

blacklist = load_forbidden_catalog("https://content.cld.iop.org/journals/0067-0049/279/2/50/revision1/apjsade2d8t3_mrt.txt")

print(f"\n🚜 STARTING HIGH-SENSITIVITY SCAN: SECTOR {SECTOR}")
print(f"   Center: RA {CENTER_RA}, Dec {CENTER_DEC}")
print(f"   Grid: {GRID_SIZE}x{GRID_SIZE} Patches.")

total_scanned = 0
patch_count = 0

for i in range(GRID_SIZE):
    current_dec = CENTER_DEC - (GRID_SIZE/2 * STEP_SIZE) + (i * STEP_SIZE)
    
    for j in range(GRID_SIZE):
        current_ra = CENTER_RA - (GRID_SIZE/2 * STEP_SIZE) + (j * STEP_SIZE)
        patch_count += 1
        
        print(f"\n📍 Patch {patch_count}/{GRID_SIZE*GRID_SIZE} | Coord: {current_ra:.1f}, {current_dec:.1f}")
        
        # RETRY LOGIC
        manifest = None
        for attempt in range(3):
            try:
                manifest = lk.search_lightcurve(
                    SkyCoord(current_ra, current_dec, unit=u.deg),
                    radius=0.7*u.deg, 
                    sector=SECTOR,
                    limit=STARS_PER_PATCH,
                    author=AUTHOR
                )
                break
            except Exception as e:
                print(f"   ⚠️ Network blip: {e}")
                time.sleep(5)
        
        if manifest is None or len(manifest) == 0:
            print("   ❌ Gap. Skipping.")
            continue
            
        print(f"   📥 Processing {len(manifest)} stars", end="")

        count = 0
        for target in manifest:
            count += 1
            if count % 20 == 0: print(".", end="", flush=True)

            try:
                tic_id = str(target.target_name).replace("TIC", "").strip()
                if tic_id in blacklist: continue 

                try:
                    lc_coll = target.download_all()
                    if not lc_coll: continue
                    lc = lc_coll.stitch().remove_nans()
                    lc = lc.normalize()
                except: continue

                # BLS
                model = lc.to_periodogram(method='bls', period=np.arange(0.3, 15.0, 0.005))
                snr = model.snr
                depth_flux = model.depth_at_max_power.value 
                period = model.period_at_max_power.value

                if depth_flux >= 1.0 or depth_flux <= 0: continue 
                mag_depth = -2.5 * np.log10(1 - depth_flux)
                
                # FILTERS
                if snr < MIN_SNR: continue 
                if mag_depth < MIN_MAG_DEPTH: continue 
                if depth_flux > MAX_DEPTH_FLUX: continue 
                if is_suspicious_period(period): continue 

                # VSX
                ra_obj = lc.meta['RA_OBJ']
                dec_obj = lc.meta['DEC_OBJ']
                is_known, name = check_vsx(ra_obj, dec_obj)
                if is_known: continue

                # SAVE
                log_candidate(tic_id, snr, mag_depth, period, ra_obj, dec_obj, "SENSITIVE_HIT")

                # PLOT
                folded = lc.fold(period=period)
                binned = folded.bin(time_bin_size=0.02)
                plt.figure(figsize=(10, 6))
                plt.plot(folded.time.value, folded.flux.value, 'k.', alpha=0.1, ms=2)
                plt.plot(binned.time.value, binned.flux.value, 'r-', lw=2, label=f"P={period:.3f}d")
                plt.title(f"TIC {tic_id} | Depth: {mag_depth:.3f}m | SNR: {snr:.1f}")
                plt.legend()
                plt.savefig(f"{OUTPUT_FOLDER}/TIC_{tic_id}.png")
                plt.close()
                
                del lc, model

            except Exception:
                continue
        
        total_scanned += len(manifest)
        print(f"\n   ✅ Patch Done. Total Scanned: {total_scanned}")
        gc.collect()

print("🏁 RASTER SCAN COMPLETE.")
