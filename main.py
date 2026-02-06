#!/usr/bin/env python3
"""
=====================================================================
DeepSkySurveyor v17 - ULTIMATE MULTI-MODE DISCOVERY ENGINE
=====================================================================

COMPREHENSIVE DISCOVERY MODES:
  1. Eclipsing Binaries (EA/EB/EW classification)
  2. Exoplanet Transits (shallow, long-period)
  3. Triple/Hierarchical Systems (multi-period)
  4. Ultra-Short Period Binaries (< 0.3 days)
  5. Heartbeat Stars (eccentric binaries)

ENHANCED FEATURES:
  - Multi-catalog cross-matching (VSX, ASAS-SN, ATLAS, Gaia)
  - Automated EB classification (EA/EB/EW)
  - Period uncertainty estimation (bootstrap)
  - Multi-period detection (finds planets in binaries, triples)
  - VSX-ready output with all required fields
  - Exoplanet validation (odd/even transit test)

USAGE:
  python DeepSkySurveyor_v17.py --mode all          # All discovery modes
  python DeepSkySurveyor_v17.py --mode eb           # Eclipsing binaries only
  python DeepSkySurveyor_v17.py --mode planet       # Exoplanets only
  python DeepSkySurveyor_v17.py --mode exotic       # Unusual systems
  python DeepSkySurveyor_v17.py --sector 5          # Specific sector

=====================================================================
"""

import sys
import os
import subprocess
import time
import json
import re
import warnings
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ===================== AUTO-INSTALL DEPENDENCIES =====================

REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'lightkurve': 'lightkurve',
    'astropy': 'astropy',
    'astroquery': 'astroquery',
    'requests': 'requests'
}

print("=" * 70)
print("  DeepSkySurveyor v17 - Ultimate Discovery Engine")
print("=" * 70)
print("\n📦 Checking dependencies...")

for module, package in REQUIRED_PACKAGES.items():
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ⏳ Installing {package}...", end=" ", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            __import__(module)
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            sys.exit(1)

print("  All dependencies ready.\n")

# Now import everything
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests

import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Catalogs, Observations
from astroquery.vizier import Vizier
from scipy.signal import find_peaks

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("lightkurve").setLevel(logging.ERROR)
logging.getLogger("astroquery").setLevel(logging.ERROR)

# ===================== DISCOVERY MODES =====================

@dataclass
class DiscoveryMode:
    """Configuration for different discovery modes."""
    name: str
    min_period: float
    max_period: float
    min_depth: float
    max_depth: float
    min_duration: float
    max_duration: float
    min_snr: float
    min_transits: int
    description: str

MODES = {
    'eb': DiscoveryMode(
        name='Eclipsing Binary',
        min_period=0.3,
        max_period=15.0,
        min_depth=0.003,
        max_depth=1.0,
        min_duration=0.01,
        max_duration=0.25,
        min_snr=7.0,
        min_transits=2,
        description='Standard eclipsing binaries'
    ),
    'planet': DiscoveryMode(
        name='Exoplanet',
        min_period=0.5,
        max_period=100.0,
        min_depth=0.0001,
        max_depth=0.05,
        min_duration=0.01,
        max_duration=0.3,
        min_snr=7.0,
        min_transits=3,
        description='Transiting exoplanets'
    ),
    'ultrashort': DiscoveryMode(
        name='Ultra-Short Binary',
        min_period=0.1,
        max_period=0.3,
        min_depth=0.002,
        max_depth=1.0,
        min_duration=0.005,
        max_duration=0.1,
        min_snr=7.0,
        min_transits=3,
        description='Ultra-short period binaries'
    ),
    'longperiod': DiscoveryMode(
        name='Long-Period Binary',
        min_period=15.0,
        max_period=60.0,
        min_depth=0.005,
        max_depth=1.0,
        min_duration=0.02,
        max_duration=0.5,
        min_snr=7.0,
        min_transits=2,
        description='Long-period eclipsing systems'
    ),
    'heartbeat': DiscoveryMode(
        name='Heartbeat Star',
        min_period=0.5,
        max_period=100.0,
        min_depth=0.001,
        max_depth=0.3,
        min_duration=0.05,
        max_duration=0.8,
        min_snr=7.0,
        min_transits=2,
        description='Eccentric binary with tidal distortion'
    )
}

# ===================== CONFIGURATION =====================

class Config:
    """Survey configuration."""
    
    WORKDIR = "DeepSkySurveyor_v17"
    ALL_SECTORS = list(range(14, 70))  # Start from Sector 14
    
    # Quality filters
    MAG_MIN = 9.0
    MAG_MAX = 14.0
    MIN_GALACTIC_LAT = 10.0
    MAX_CONTAMINATION = 0.10
    MIN_DATA_POINTS = 1000
    
    # Rate limiting
    QUERY_DELAY = 1.0
    DOWNLOAD_DELAY = 1.5
    MAX_RETRIES = 3
    CHECKPOINT_EVERY = 25
    
    # Multi-period detection
    MAX_PERIODS_TO_FIND = 3
    MASKING_PHASE_WIDTH = 0.15
    
    # Bootstrap uncertainty
    N_BOOTSTRAP = 50

# ===================== DATA STRUCTURES =====================

@dataclass
class CatalogMatch:
    """Cross-match with external catalogs."""
    vsx: Optional[str] = None
    asassn: Optional[str] = None
    atlas: Optional[str] = None
    gaia_var: Optional[str] = None

# ===================== FILE MANAGEMENT =====================

class FileManager:
    def __init__(self, sector=None):
        self.root = Path(Config.WORKDIR)
        self.sector = sector
        
        if sector:
            self.sector_dir = self.root / f"Sector_{sector:02d}"
        else:
            self.sector_dir = self.root / "auto"
            
        self.dirs = {
            'plots': self.sector_dir / 'plots',
            'plots_multi': self.sector_dir / 'plots_multi',
            'cache': self.root / 'cache',
        }
        self.files = {
            'targets': self.sector_dir / 'targets.csv',
            'progress': self.sector_dir / 'progress.json',
            'candidates_eb': self.sector_dir / 'candidates_eb.csv',
            'candidates_planet': self.sector_dir / 'candidates_planet.csv',
            'candidates_exotic': self.sector_dir / 'candidates_exotic.csv',
            'candidates_all': self.sector_dir / 'candidates_all.csv',
            'vsx_ready': self.sector_dir / 'vsx_submissions.csv',
            'rejected': self.sector_dir / 'rejected.csv',
            'kostov': self.root / 'cache' / 'kostov_ebs.txt',
            'sectors_log': self.root / 'processed_sectors.json',
        }
        self._create()

    def _create(self):
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sector_dir.mkdir(parents=True, exist_ok=True)

# ===================== PROGRESS TRACKING =====================

class Progress:
    def __init__(self, filepath):
        self.file = filepath
        self.data = self._load()
        self.start = time.time()
        self.times = []

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file) as f:
                    return json.load(f)
            except:
                pass
        return {'processed': [], 'detections': [], 'rejected': []}

    def save(self):
        self.data['last_update'] = datetime.now().isoformat()
        with open(self.file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_done(self, tic):
        return str(tic) in self.data['processed']

    def mark_done(self, tic, t=None):
        if str(tic) not in self.data['processed']:
            self.data['processed'].append(str(tic))
        if t:
            self.times.append(t)
            self.times = self.times[-100:]

    def add_detection(self, tic, detection):
        self.data['detections'].append({'tic': str(tic), 'data': detection})

    def add_rejected(self, tic, reason):
        self.data['rejected'].append({'tic': str(tic), 'reason': reason})

    def eta(self, remaining):
        if not self.times:
            return "calculating..."
        return str(timedelta(seconds=int(np.mean(self.times) * remaining)))

    def rate(self):
        elapsed = time.time() - self.start
        if elapsed > 0 and len(self.times) > 0:
            return len(self.times) / (elapsed / 3600)
        return 0

# ===================== KNOWN CATALOGS =====================

class KnownCatalogs:
    """Manage all known variable star catalogs."""
    
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.kostov = self._load_kostov()
        
    def _load_kostov(self):
        """Load Kostov EB catalog."""
        cache_file = self.cache_dir / 'kostov_ebs.txt'
        known = set()
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    known = set(l.strip() for l in f if l.strip())
                if len(known) > 1000:
                    print(f"  ✓ {len(known)} Kostov EBs loaded")
                    return known
            except:
                pass

        print("  📥 Downloading Kostov catalog...", end=" ", flush=True)
        try:
            url = "https://content.cld.iop.org/journals/0067-0049/279/2/50/revision1/apjsade2d8t3_mrt.txt"
            r = requests.get(url, timeout=60)
            for line in r.text.split('\n'):
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    known.add(parts[0])
            with open(cache_file, 'w') as f:
                f.write('\n'.join(sorted(known)))
            print(f"{len(known)} EBs")
        except Exception as e:
            print(f"failed: {e}")
        
        return known
    
    def is_known_kostov(self, tic):
        return str(tic) in self.kostov

# ===================== CATALOG CROSS-MATCHING =====================

class CatalogMatcher:
    """Cross-match with all major catalogs."""

    @staticmethod
    def get_tic_info(tic):
        """Get comprehensive TIC catalog info."""
        try:
            result = Catalogs.query_criteria(catalog="TIC", ID=tic)
            if not result or len(result) == 0:
                return None

            row = result[0]

            def get(key, default=None):
                try:
                    v = row[key]
                    if v is None or np.ma.is_masked(v):
                        return default
                    return float(v)
                except:
                    return default

            ra, dec = get('ra'), get('dec')
            gal_lat = None
            if ra and dec:
                try:
                    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                    gal_lat = c.galactic.b.deg
                except:
                    pass

            return {
                'tic': str(tic),
                'ra': ra,
                'dec': dec,
                'tmag': get('Tmag'),
                'teff': get('Teff'),
                'radius': get('rad'),
                'mass': get('mass'),
                'logg': get('logg'),
                'gal_lat': gal_lat,
                'contamination': get('contratio', 0),
                'parallax': get('plx')
            }
        except:
            return None

    @staticmethod
    def check_all_catalogs(ra, dec, radius=30) -> CatalogMatch:
        """Check all major variable star catalogs."""
        matches = CatalogMatch()
        
        try:
            c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            
            # VSX
            try:
                v = Vizier(columns=['Name', 'Type'], row_limit=3)
                r = v.query_region(c, radius=radius*u.arcsec, catalog='B/vsx/vsx')
                if r and len(r) > 0 and len(r[0]) > 0:
                    matches.vsx = f"{r[0][0]['Name']} ({r[0][0]['Type']})"
            except:
                pass
            
            # ASAS-SN Variable Stars
            try:
                v = Vizier(columns=['Name', 'Type', 'Period'], row_limit=3)
                r = v.query_region(c, radius=radius*u.arcsec, catalog='II/366/catalog')
                if r and len(r) > 0 and len(r[0]) > 0:
                    matches.asassn = f"{r[0][0]['Name']}"
            except:
                pass
            
            # ATLAS Variables
            try:
                v = Vizier(columns=['ATLAS', 'Type'], row_limit=3)
                r = v.query_region(c, radius=radius*u.arcsec, catalog='J/AJ/156/241')
                if r and len(r) > 0 and len(r[0]) > 0:
                    matches.atlas = f"ATLAS-{r[0][0]['ATLAS']}"
            except:
                pass
            
        except:
            pass
        
        return matches

# ===================== SECTOR MANAGER =====================

class SectorManager:
    """Tracks which sectors have been processed."""
    
    def __init__(self, sectors_file):
        self.file = sectors_file
        self.data = self._load()
    
    def _load(self):
        if self.file.exists():
            try:
                with open(self.file) as f:
                    return json.load(f)
            except:
                pass
        return {'completed': [], 'in_progress': None}
    
    def save(self):
        with open(self.file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_next_sector(self):
        completed = set(self.data.get('completed', []))
        for sector in Config.ALL_SECTORS:
            if sector not in completed:
                return sector
        return None
    
    def mark_complete(self, sector):
        if sector not in self.data.get('completed', []):
            self.data['completed'].append(sector)
            self.data['completed'].sort()
        self.data['in_progress'] = None
        self.save()
    
    def mark_in_progress(self, sector):
        self.data['in_progress'] = sector
        self.save()

# ===================== TARGET FINDER =====================

class TargetFinder:
    """Get ALL targets from a sector."""

    def __init__(self, fm, catalogs, sector):
        self.fm = fm
        self.catalogs = catalogs
        self.sector = sector

    def get_targets(self):
        """Get all targets from the specified sector."""
        if self.fm.files['targets'].exists():
            try:
                df = pd.read_csv(self.fm.files['targets'])
                if len(df) >= 10:
                    targets = df['tic'].astype(str).tolist()
                    print(f"  ✓ Loaded {len(targets)} cached targets for Sector {self.sector}")
                    return targets
            except:
                pass

        print(f"\n🔍 Acquiring ALL targets from Sector {self.sector}...")
        all_tics = self._query_mast()
        
        if not all_tics:
            print(f"\n  ❌ No targets found in Sector {self.sector}")
            return []

        # Filter known Kostov EBs
        novel = [t for t in all_tics if not self.catalogs.is_known_kostov(t)]
        
        print(f"\n  ✅ Sector {self.sector}: {len(novel)} targets")
        print(f"     (Filtered {len(all_tics) - len(novel)} known Kostov EBs)")

        df = pd.DataFrame({'tic': novel, 'sector': self.sector})
        df.to_csv(self.fm.files['targets'], index=False)

        return novel

    def _query_mast(self):
        """Query MAST for all observations in this sector."""
        print(f"  📡 Querying MAST for Sector {self.sector}...")
        
        all_tics = set()
        
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                dataproduct_type="timeseries",
                sequence_number=self.sector
            )

            if obs and len(obs) > 0:
                print(f"      Found {len(obs)} observations")
                
                for target_name in obs['target_name']:
                    match = re.search(r'TIC[\s-]?(\d{8,})', str(target_name), re.IGNORECASE)
                    if match:
                        all_tics.add(match.group(1))
                    else:
                        match = re.search(r'(\d{8,})', str(target_name))
                        if match:
                            all_tics.add(match.group(1))

                print(f"      Extracted {len(all_tics)} unique TIC IDs")

        except Exception as e:
            print(f"      MAST Error: {str(e)[:60]}")

        return list(all_tics)

# ===================== ADVANCED ANALYZER =====================

class AdvancedAnalyzer:
    """Multi-mode detection with classification and validation."""

    def __init__(self, fm):
        self.fm = fm

    def analyze_all_modes(self, tic, modes=['eb', 'planet', 'ultrashort', 'longperiod']):
        """Analyze with multiple detection modes."""
        
        # Download light curve
        lc, n_sectors = self._download(tic)
        if lc is None:
            return {'status': 'rejected', 'reason': 'No light curve data'}

        if len(lc) < Config.MIN_DATA_POINTS:
            return {'status': 'rejected', 'reason': f'Insufficient data ({len(lc)} pts)'}

        # Process
        try:
            lc_clean = lc.remove_nans().remove_outliers(sigma=5)
            lc_flat = lc_clean.flatten(window_length=501)
        except:
            return {'status': 'rejected', 'reason': 'Processing failed'}

        detections = []
        
        # Try each mode
        for mode_name in modes:
            if mode_name not in MODES:
                continue
                
            mode = MODES[mode_name]
            result = self._detect_with_mode(lc_flat, mode, tic)
            
            if result and result['status'] == 'detected':
                detections.append(result)
        
        if not detections:
            return {'status': 'rejected', 'reason': 'No signals detected'}
        
        # Multi-period search on best detection
        best = max(detections, key=lambda x: x['snr'])
        multi_periods = self._find_multiple_periods(lc_flat, best)
        
        if len(multi_periods) > 1:
            best['multi_period'] = True
            best['all_periods'] = multi_periods
            best['classification'] = 'Triple/Hierarchical System'
        
        best['lc'] = lc_flat
        best['n_sectors'] = n_sectors
        best['n_points'] = len(lc_flat)
        
        return best

    def _detect_with_mode(self, lc, mode: DiscoveryMode, tic):
        """Detect signal using specific mode parameters."""
        
        # Run BLS
        bls_result = self._run_bls(lc, mode)
        if bls_result is None:
            return None

        # Validate
        validation = self._validate(bls_result, lc, mode)
        if not validation['valid']:
            return None

        # Classify if EB mode
        classification = 'Unknown'
        secondary_depth = 0.0
        eccentricity = 0.0
        
        if mode.name == 'Eclipsing Binary':
            eb_class = self._classify_eb(lc, bls_result['period'], bls_result['epoch'])
            classification = eb_class['type']
            secondary_depth = eb_class['secondary_depth']
        
        # Check if heartbeat
        if mode.name == 'Heartbeat Star':
            is_heartbeat, ecc = self._check_heartbeat(lc, bls_result['period'], bls_result['epoch'])
            if is_heartbeat:
                eccentricity = ecc
                classification = 'Heartbeat Star'
        
        # Estimate period uncertainty
        period_err = self._bootstrap_period(lc, bls_result['period'], bls_result['epoch'])
        
        # Calculate magnitude range
        mag_range = self._calculate_mag_range(lc, bls_result['depth'])
        
        # Planet probability (for planet mode)
        prob_planet = 0.0
        if mode.name == 'Exoplanet':
            prob_planet = self._calculate_planet_probability(lc, bls_result)
        
        return {
            'status': 'detected',
            'tic': str(tic),
            'mode': mode.name,
            'period': bls_result['period'],
            'period_err': period_err,
            'epoch': bls_result['epoch'],
            'depth': bls_result['depth'],
            'depth_pct': bls_result['depth'] * 100,
            'duration': bls_result['duration'],
            'snr': bls_result['snr'],
            'n_transits': validation['n_transits'],
            'classification': classification,
            'secondary_depth': secondary_depth,
            'eccentricity': eccentricity,
            'prob_planet': prob_planet,
            'mag_range': mag_range,
            'multi_period': False
        }

    def _run_bls(self, lc, mode: DiscoveryMode):
        """Run BLS with mode-specific parameters."""
        try:
            t = lc.time.value
            y = lc.flux.value

            periods = np.linspace(mode.min_period, mode.max_period, 5000)
            
            # Adaptive duration grid
            max_dur = min(mode.max_duration, mode.min_period * 0.8)
            durations = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25])
            durations = durations[(durations >= mode.min_duration) & (durations <= max_dur)]
            
            if len(durations) == 0:
                durations = np.array([mode.min_duration, mode.min_duration * 2])

            bls = BoxLeastSquares(t, y)
            periodogram = bls.power(periods, duration=durations)

            idx = np.argmax(periodogram.power)
            best_period = periodogram.period[idx]
            best_duration = periodogram.duration[idx]
            best_t0 = periodogram.transit_time[idx]

            stats = bls.compute_stats(best_period, best_duration, best_t0)
            depth = abs(stats['depth'][0])

            noise = np.std(y)
            snr = (depth / noise) * np.sqrt(len(t) * best_duration / best_period)

            return {
                'period': float(best_period),
                'epoch': float(best_t0),
                'duration': float(best_duration),
                'depth': float(depth),
                'snr': float(snr),
                'power': float(periodogram.power[idx])
            }

        except Exception as e:
            return None

    def _validate(self, bls, lc, mode: DiscoveryMode):
        """Validate detection with mode-specific criteria."""

        if bls['snr'] < mode.min_snr:
            return {'valid': False, 'reason': f"Low SNR ({bls['snr']:.1f})"}

        if bls['depth'] < mode.min_depth or bls['depth'] > mode.max_depth:
            return {'valid': False, 'reason': f"Depth out of range"}

        p = bls['period']

        # Instrumental aliases
        for n in [1, 2, 3, 4]:
            if abs(p - 0.99727/n) / (0.99727/n) < 0.01:
                return {'valid': False, 'reason': 'Sidereal alias'}

        for d in [0.5, 1.0, 2.0]:
            if abs(p - d) < 0.02:
                return {'valid': False, 'reason': 'Day alias'}

        if abs(p - 13.7) / 13.7 < 0.02 or abs(p - 6.85) / 6.85 < 0.02:
            return {'valid': False, 'reason': 'TESS orbit alias'}

        # Count transits
        t = lc.time.value
        phase = ((t - bls['epoch']) % p) / p
        in_transit = (phase < 0.1) | (phase > 0.9)
        n_transits = max(1, np.sum(np.diff(in_transit.astype(int)) == 1))

        if n_transits < mode.min_transits:
            return {'valid': False, 'reason': f'Few transits ({n_transits})'}

        return {'valid': True, 'n_transits': n_transits}

    def _classify_eb(self, lc, period, epoch):
        """Classify eclipsing binary as EA/EB/EW."""
        try:
            folded = lc.fold(period=period, epoch_time=epoch)
            
            # Primary eclipse depth
            primary_mask = np.abs(folded.phase.value) < 0.1
            primary_flux = folded.flux.value[primary_mask]
            primary_depth = 1.0 - np.median(primary_flux) if len(primary_flux) > 10 else 0
            
            # Secondary eclipse depth
            secondary_mask = (folded.phase.value > 0.4) & (folded.phase.value < 0.6)
            secondary_flux = folded.flux.value[secondary_mask]
            secondary_depth = 1.0 - np.median(secondary_flux) if len(secondary_flux) > 10 else 0
            
            # Out-of-eclipse flux
            oot_mask = (folded.phase.value > 0.2) & (folded.phase.value < 0.3)
            oot_flux = folded.flux.value[oot_mask]
            oot_level = np.median(oot_flux) if len(oot_flux) > 10 else 1.0
            
            # Normalize depths
            primary_depth = abs(primary_depth / oot_level) if oot_level > 0 else 0
            secondary_depth = abs(secondary_depth / oot_level) if oot_level > 0 else 0
            
            # Classification criteria
            depth_ratio = secondary_depth / primary_depth if primary_depth > 0 else 0
            
            if depth_ratio > 0.8:
                eb_type = 'EW'  # Contact binary
            elif depth_ratio > 0.1:
                eb_type = 'EB'  # Semi-detached
            else:
                eb_type = 'EA'  # Detached
            
            return {
                'type': eb_type,
                'secondary_depth': float(secondary_depth),
                'depth_ratio': float(depth_ratio)
            }
            
        except:
            return {'type': 'EA', 'secondary_depth': 0.0, 'depth_ratio': 0.0}

    def _check_heartbeat(self, lc, period, epoch):
        """Check if system shows heartbeat morphology."""
        try:
            folded = lc.fold(period=period, epoch_time=epoch)
            
            # Look for characteristic asymmetric peak
            phase_bins = np.linspace(-0.5, 0.5, 200)
            binned_flux = []
            
            for i in range(len(phase_bins) - 1):
                mask = (folded.phase.value >= phase_bins[i]) & (folded.phase.value < phase_bins[i+1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.median(folded.flux.value[mask]))
                else:
                    binned_flux.append(np.nan)
            
            binned_flux = np.array(binned_flux)
            
            # Find peaks (heartbeat characteristic)
            peaks, properties = find_peaks(binned_flux, prominence=0.002)
            
            # Heartbeat shows sharp, asymmetric peak near periastron
            if len(peaks) > 0:
                # Check for asymmetry
                peak_idx = peaks[0]
                if peak_idx > 10 and peak_idx < len(binned_flux) - 10:
                    left_slope = binned_flux[peak_idx] - binned_flux[peak_idx - 5]
                    right_slope = binned_flux[peak_idx] - binned_flux[peak_idx + 5]
                    
                    # Asymmetry indicator
                    asymmetry = abs(left_slope - right_slope) / max(abs(left_slope), abs(right_slope), 1e-6)
                    
                    if asymmetry > 0.3:
                        # Estimate eccentricity from peak width
                        ecc = min(0.9, 0.1 + asymmetry)
                        return True, ecc
            
            return False, 0.0
            
        except:
            return False, 0.0

    def _bootstrap_period(self, lc, period, epoch, n_bootstrap=50):
        """Estimate period uncertainty via bootstrap."""
        try:
            t = lc.time.value
            y = lc.flux.value
            n = len(t)
            
            periods = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n, size=n, replace=True)
                t_boot = t[indices]
                y_boot = y[indices]
                
                # Sort by time
                sort_idx = np.argsort(t_boot)
                t_boot = t_boot[sort_idx]
                y_boot = y_boot[sort_idx]
                
                # Run BLS on bootstrap sample
                try:
                    p_range = np.linspace(period * 0.95, period * 1.05, 200)
                    bls = BoxLeastSquares(t_boot, y_boot)
                    periodogram = bls.power(p_range, duration=0.05)
                    periods.append(p_range[np.argmax(periodogram.power)])
                except:
                    continue
            
            if len(periods) > 10:
                return float(np.std(periods))
            else:
                return 0.0
                
        except:
            return 0.0

    def _calculate_mag_range(self, lc, depth):
        """Calculate magnitude range from depth."""
        # Approximate - assumes small depth
        delta_mag = 2.5 * np.log10(1.0 / (1.0 - depth)) if depth < 0.5 else 0.5
        return (0.0, float(delta_mag))

    def _calculate_planet_probability(self, lc, bls_result):
        """Calculate probability that signal is a planet."""
        try:
            folded = lc.fold(period=bls_result['period'], epoch_time=bls_result['epoch'])
            
            # Odd/even transit test
            t = lc.time.value
            period = bls_result['period']
            epoch = bls_result['epoch']
            
            # Separate odd and even transits
            transit_nums = np.round((t - epoch) / period)
            odd_mask = (transit_nums % 2 == 1) & (np.abs(folded.phase.value) < 0.1)
            even_mask = (transit_nums % 2 == 0) & (np.abs(folded.phase.value) < 0.1)
            
            if np.sum(odd_mask) > 10 and np.sum(even_mask) > 10:
                odd_depth = 1.0 - np.median(lc.flux.value[odd_mask])
                even_depth = 1.0 - np.median(lc.flux.value[even_mask])
                
                depth_diff = abs(odd_depth - even_depth)
                avg_depth = (odd_depth + even_depth) / 2
                
                # Planets have consistent depths, binaries often don't
                consistency = 1.0 - min(1.0, depth_diff / avg_depth) if avg_depth > 0 else 0.5
                
                # Small depth favors planet
                depth_score = 1.0 if bls_result['depth'] < 0.02 else 0.5
                
                prob = (consistency * 0.6 + depth_score * 0.4)
                return min(1.0, max(0.0, float(prob)))
            
            return 0.5
            
        except:
            return 0.5

    def _find_multiple_periods(self, lc, primary_detection):
        """Search for additional periods after masking primary."""
        try:
            periods_found = [primary_detection['period']]
            
            # Work with copy
            t = lc.time.value.copy()
            y = lc.flux.value.copy()
            
            # Mask primary transits
            phase = ((t - primary_detection['epoch']) % primary_detection['period']) / primary_detection['period']
            mask_width = Config.MASKING_PHASE_WIDTH
            in_transit = (phase < mask_width) | (phase > (1 - mask_width))
            
            # Set in-transit points to median
            y[in_transit] = np.median(y[~in_transit])
            
            # Search for secondary period
            for attempt in range(Config.MAX_PERIODS_TO_FIND - 1):
                # Run BLS on masked data
                periods = np.linspace(0.3, 30.0, 3000)
                durations = np.array([0.01, 0.02, 0.05, 0.10])
                
                bls = BoxLeastSquares(t, y)
                periodogram = bls.power(periods, duration=durations)
                
                idx = np.argmax(periodogram.power)
                new_period = periodogram.period[idx]
                power = periodogram.power[idx]
                
                # Check if significant
                if power < 0.05:  # Arbitrary threshold
                    break
                
                # Check if harmonically related
                is_harmonic = False
                for p in periods_found:
                    for n in [0.5, 2.0, 3.0, 0.333, 1.5]:
                        if abs(new_period - p * n) / (p * n) < 0.05:
                            is_harmonic = True
                            break
                
                if not is_harmonic:
                    periods_found.append(float(new_period))
                    
                    # Mask this period too
                    phase2 = ((t - periodogram.transit_time[idx]) % new_period) / new_period
                    in_transit2 = (phase2 < mask_width) | (phase2 > (1 - mask_width))
                    y[in_transit2] = np.median(y[~(in_transit | in_transit2)])
                else:
                    break
            
            return periods_found
            
        except:
            return [primary_detection['period']]

    def _download(self, tic):
        """Download light curve."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                search = lk.search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
                if len(search) == 0:
                    search = lk.search_lightcurve(f"TIC {tic}", author="SPOC")
                if len(search) == 0:
                    return None, 0

                lc_coll = search.download_all()
                if not lc_coll or len(lc_coll) == 0:
                    return None, 0

                return lc_coll.stitch(), len(lc_coll)

            except:
                time.sleep(Config.DOWNLOAD_DELAY)
                continue

        return None, 0

# ===================== ENHANCED PLOTTER =====================

class EnhancedPlotter:
    def __init__(self, fm):
        self.fm = fm

    def plot_detection(self, tic, detection, lc, catalog_info):
        """Create comprehensive diagnostic plot."""
        try:
            # Create figure with custom layout
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Title with classification
            fig.suptitle(f"TIC {tic} - {detection['classification']}\n"
                        f"P={detection['period']:.6f}±{detection['period_err']:.6f} d, "
                        f"Depth={detection['depth_pct']:.3f}%, SNR={detection['snr']:.1f}",
                        fontsize=14, fontweight='bold')
            
            # 1. Full light curve
            ax1 = fig.add_subplot(gs[0, :])
            ax1.scatter(lc.time.value, lc.flux.value, s=0.5, alpha=0.4, c='black')
            ax1.set_xlabel('Time (BTJD)')
            ax1.set_ylabel('Normalized Flux')
            ax1.set_title(f'{detection["n_points"]} points, {detection["n_sectors"]} sectors')
            ax1.grid(alpha=0.3)
            
            # 2. Phase-folded
            ax2 = fig.add_subplot(gs[1, 0])
            folded = lc.fold(period=detection['period'], epoch_time=detection['epoch'])
            ax2.scatter(folded.phase.value, folded.flux.value, s=0.8, alpha=0.3, c='navy')
            
            # Binned
            bins = np.linspace(-0.5, 0.5, 100)
            binned = []
            for i in range(len(bins)-1):
                mask = (folded.phase.value >= bins[i]) & (folded.phase.value < bins[i+1])
                if np.sum(mask) > 0:
                    binned.append(np.median(folded.flux.value[mask]))
                else:
                    binned.append(np.nan)
            ax2.plot((bins[:-1]+bins[1:])/2, binned, 'r-', lw=2, label='Binned')
            
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Flux')
            ax2.set_title(f'Full Phase ({detection["n_transits"]} transits)')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # 3. Primary eclipse zoom
            ax3 = fig.add_subplot(gs[1, 1])
            mask = np.abs(folded.phase.value) < 0.15
            if np.sum(mask) > 10:
                ax3.scatter(folded.phase.value[mask], folded.flux.value[mask], 
                           s=2, c='red', alpha=0.6)
            ax3.set_xlabel('Phase')
            ax3.set_ylabel('Flux')
            ax3.set_title(f'Primary (Depth {detection["depth_pct"]:.3f}%)')
            ax3.set_xlim(-0.15, 0.15)
            ax3.grid(alpha=0.3)
            
            # 4. Secondary eclipse zoom
            ax4 = fig.add_subplot(gs[1, 2])
            mask = (folded.phase.value > 0.35) & (folded.phase.value < 0.65)
            if np.sum(mask) > 10:
                ax4.scatter(folded.phase.value[mask], folded.flux.value[mask], 
                           s=2, c='green', alpha=0.6)
            ax4.set_xlabel('Phase')
            ax4.set_ylabel('Flux')
            sec_depth = detection.get('secondary_depth', 0)
            ax4.set_title(f'Secondary (Depth {sec_depth*100:.3f}%)')
            ax4.set_xlim(0.35, 0.65)
            ax4.grid(alpha=0.3)
            
            # 5. Information panel
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            
            info_text = f"""
DETECTION INFORMATION:
├─ Classification: {detection['classification']}
├─ Period: {detection['period']:.7f} ± {detection['period_err']:.7f} days
├─ Epoch (T0): {detection['epoch']:.5f} BTJD
├─ Primary Depth: {detection['depth_pct']:.4f}%
├─ Secondary Depth: {detection.get('secondary_depth', 0)*100:.4f}%
├─ Duration: {detection['duration']*24:.2f} hours
├─ SNR: {detection['snr']:.2f}
└─ Transits Observed: {detection['n_transits']}

STELLAR PARAMETERS (TIC):
├─ RA/Dec: {catalog_info.get('ra', 0):.4f}, {catalog_info.get('dec', 0):.4f}
├─ Tmag: {catalog_info.get('tmag', 0):.2f}
├─ Teff: {catalog_info.get('teff', 0):.0f} K
├─ Radius: {catalog_info.get('radius', 0):.2f} R☉
└─ Mass: {catalog_info.get('mass', 0):.2f} M☉
"""
            
            if detection.get('multi_period', False):
                info_text += f"\nMULTI-PERIOD SYSTEM:\n"
                for i, p in enumerate(detection.get('all_periods', []), 1):
                    info_text += f"  Period {i}: {p:.6f} d\n"
            
            if detection.get('eccentricity', 0) > 0:
                info_text += f"\nEccentricity: {detection['eccentricity']:.3f}"
            
            if detection.get('prob_planet', 0) > 0:
                info_text += f"\nPlanet Probability: {detection['prob_planet']*100:.1f}%"
            
            ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            # Save
            plot_dir = self.fm.dirs['plots_multi'] if detection.get('multi_period') else self.fm.dirs['plots']
            path = plot_dir / f"TIC_{tic}_{detection['mode'].replace(' ', '_')}.png"
            plt.savefig(path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return path.name
            
        except Exception as e:
            print(f"Plot error: {e}")
            return None

# ===================== MAIN SURVEY =====================

class EnhancedSurvey:
    def __init__(self, sector=None, modes=['eb', 'planet', 'ultrashort']):
        print("\n" + "=" * 70)
        print("  ENHANCED MULTI-MODE DISCOVERY SURVEY")
        print("=" * 70)

        # Create root directory first
        Path(Config.WORKDIR).mkdir(parents=True, exist_ok=True)

        # Determine sector
        sector_mgr = SectorManager(Path(Config.WORKDIR) / 'processed_sectors.json')
        
        if sector is None:
            sector = sector_mgr.get_next_sector()
            if sector is None:
                print("\n🎉 ALL SECTORS COMPLETE!")
                sys.exit(0)
            print(f"\n  Auto-selected: Sector {sector}")
        else:
            print(f"\n  User-specified: Sector {sector}")
        
        sector_mgr.mark_in_progress(sector)
        
        self.sector = sector
        self.sector_mgr = sector_mgr
        self.modes = modes
        self.fm = FileManager(sector)
        self.catalogs = KnownCatalogs(self.fm.dirs['cache'])
        self.progress = Progress(self.fm.files['progress'])
        self.finder = TargetFinder(self.fm, self.catalogs, sector)
        self.analyzer = AdvancedAnalyzer(self.fm)
        self.plotter = EnhancedPlotter(self.fm)
        
        self.detections = {
            'eb': [],
            'planet': [],
            'exotic': []
        }

    def run(self):
        """Run full enhanced survey."""

        # Get targets
        targets = self.finder.get_targets()
        if not targets:
            print(f"\n❌ No targets for Sector {self.sector}")
            print(f"   Moving to next sector...")
            self.sector_mgr.mark_complete(self.sector)
            
            # Try next sector automatically
            next_sector = self.sector_mgr.get_next_sector()
            if next_sector:
                print(f"\n🔄 Auto-switching to Sector {next_sector}...")
                time.sleep(2)
                EnhancedSurvey(sector=next_sector, modes=self.modes).run()
            return

        # Filter processed
        remaining = [t for t in targets if not self.progress.is_done(t)]

        print(f"\n📊 Sector {self.sector} Status:")
        print(f"   Total targets: {len(targets)}")
        print(f"   Already done: {len(targets) - len(remaining)}")
        print(f"   Remaining: {len(remaining)}")
        print(f"   Detection modes: {', '.join(self.modes)}")
        print(f"   Est. runtime: {len(remaining) * 8 / 3600:.1f} hours")

        if not remaining:
            print(f"\n✅ Sector {self.sector} complete!")
            self.sector_mgr.mark_complete(self.sector)
            self._report()
            
            # Check if we found anything
            if len(self.progress.data['detections']) == 0:
                print(f"\n⚠️  No detections in Sector {self.sector}. Moving to next sector...")
                next_sector = self.sector_mgr.get_next_sector()
                if next_sector:
                    print(f"🔄 Auto-switching to Sector {next_sector}...")
                    time.sleep(2)
                    EnhancedSurvey(sector=next_sector, modes=self.modes).run()
            return

        print(f"\n{'='*70}")
        print(f"  PROCESSING SECTOR {self.sector}")
        print("=" * 70 + "\n")

        # Process
        try:
            for i, tic in enumerate(remaining):
                self._process(tic, i+1, len(remaining))

                if (i+1) % Config.CHECKPOINT_EVERY == 0:
                    self.progress.save()
                    self._save_detections()

            # Complete
            self.progress.save()
            self._save_detections()
            self._generate_vsx_submissions()
            self._report()
            self.sector_mgr.mark_complete(self.sector)

            print("\n" + "=" * 70)
            print(f"  SECTOR {self.sector} COMPLETE")
            print("=" * 70)
            print(f"   Processed: {len(self.progress.data['processed'])}")
            print(f"   Detections: {len(self.progress.data['detections'])}")
            print(f"   Results: {self.fm.sector_dir}/")
            print("=" * 70 + "\n")
            
            # Auto-advance if no detections found
            if len(self.progress.data['detections']) == 0:
                print(f"\n⚠️  No detections in Sector {self.sector}. Moving to next sector...")
                next_sector = self.sector_mgr.get_next_sector()
                if next_sector:
                    print(f"🔄 Auto-switching to Sector {next_sector}...")
                    time.sleep(2)
                    EnhancedSurvey(sector=next_sector, modes=self.modes).run()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Progress saved.\n")
            self.progress.save()
            self._save_detections()

    def _process(self, tic, current, total):
        """Process single target with all modes."""
        t0 = time.time()

        eta = self.progress.eta(total - current)
        rate = self.progress.rate()
        print(f"[{current}/{total}] TIC {tic} (ETA: {eta}, {rate:.0f}/hr)...", end=" ", flush=True)

        # Get catalog info
        info = CatalogMatcher.get_tic_info(tic)
        if not info or not info['ra']:
            print("❌ No TIC info")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, "No TIC info")
            return

        # Quality filters
        if info['tmag'] and (info['tmag'] < Config.MAG_MIN or info['tmag'] > Config.MAG_MAX):
            print(f"❌ Tmag={info['tmag']:.1f}")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, f"Tmag {info['tmag']:.1f}")
            return

        if info['gal_lat'] and abs(info['gal_lat']) < Config.MIN_GALACTIC_LAT:
            print(f"❌ |b|={abs(info['gal_lat']):.1f}°")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, f"Galactic lat")
            return

        if info['contamination'] and info['contamination'] > Config.MAX_CONTAMINATION:
            print(f"❌ Contam={info['contamination']*100:.0f}%")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, "Contamination")
            return

        # Cross-match catalogs
        matches = CatalogMatcher.check_all_catalogs(info['ra'], info['dec'])
        if matches.vsx or matches.asassn:
            known = matches.vsx or matches.asassn
            print(f"❌ Known: {known[:30]}")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, f"Known: {known}")
            return

        # Analyze with all modes
        result = self.analyzer.analyze_all_modes(tic, self.modes)

        if result['status'] == 'rejected':
            print(f"❌ {result['reason']}")
            self.progress.mark_done(tic, time.time()-t0)
            self.progress.add_rejected(tic, result['reason'])
            return

        # DETECTION!
        det_type = "🌟" if result.get('multi_period') else "✅"
        print(f"{det_type} {result['classification']} P={result['period']:.4f}d "
              f"D={result['depth_pct']:.2f}% SNR={result['snr']:.0f}")

        # Plot
        plot = self.plotter.plot_detection(tic, result, result['lc'], info)

        # Store detection
        detection_data = {
            'TIC': tic,
            'Sector': self.sector,
            'RA': info['ra'],
            'Dec': info['dec'],
            'Tmag': info['tmag'],
            'Teff': info['teff'],
            'Radius': info['radius'],
            'Mass': info['mass'],
            'Mode': result['mode'],
            'Classification': result['classification'],
            'Period_d': result['period'],
            'Period_err_d': result['period_err'],
            'Epoch_BTJD': result['epoch'],
            'Depth_pct': result['depth_pct'],
            'Secondary_Depth_pct': result.get('secondary_depth', 0) * 100,
            'Duration_hr': result['duration'] * 24,
            'SNR': result['snr'],
            'N_transits': result['n_transits'],
            'N_sectors': result['n_sectors'],
            'N_points': result['n_points'],
            'Eccentricity': result.get('eccentricity', 0),
            'Planet_Prob': result.get('prob_planet', 0),
            'Multi_Period': result.get('multi_period', False),
            'Plot': plot,
            'Catalog_Matches': str(matches)
        }

        # Categorize
        if result['mode'] in ['Eclipsing Binary', 'Ultra-Short Binary']:
            self.detections['eb'].append(detection_data)
        elif result['mode'] == 'Exoplanet':
            self.detections['planet'].append(detection_data)
        else:
            self.detections['exotic'].append(detection_data)

        self.progress.mark_done(tic, time.time()-t0)
        self.progress.add_detection(tic, detection_data)

        time.sleep(Config.DOWNLOAD_DELAY)

    def _save_detections(self):
        """Save all detections to CSV files."""
        if self.detections['eb']:
            pd.DataFrame(self.detections['eb']).to_csv(
                self.fm.files['candidates_eb'], index=False)
        if self.detections['planet']:
            pd.DataFrame(self.detections['planet']).to_csv(
                self.fm.files['candidates_planet'], index=False)
        if self.detections['exotic']:
            pd.DataFrame(self.detections['exotic']).to_csv(
                self.fm.files['candidates_exotic'], index=False)
        
        # Combined
        all_dets = self.detections['eb'] + self.detections['planet'] + self.detections['exotic']
        if all_dets:
            pd.DataFrame(all_dets).to_csv(
                self.fm.files['candidates_all'], index=False)

    def _generate_vsx_submissions(self):
        """Generate VSX-ready submission file."""
        vsx_ready = []
        
        for det in self.detections['eb']:
            mag_min, mag_max = det['Tmag'], det['Tmag'] + (det['Depth_pct'] / 100 * 2.5)
            
            vsx_entry = {
                'Name': f"TIC {det['TIC']}",
                'RA_J2000': det['RA'],
                'Dec_J2000': det['Dec'],
                'Type': det['Classification'],
                'Max_Mag': f"{mag_min:.2f}",
                'Min_Mag': f"{mag_max:.2f}",
                'Photometric_System': 'TESS',
                'Epoch': f"{det['Epoch_BTJD']:.5f}",
                'Period_days': f"{det['Period_d']:.7f}",
                'Period_Uncertainty': f"{det['Period_err_d']:.7f}",
                'Discoverer': 'DeepSkySurveyor',
                'Remarks': f"Detected in TESS Sector {det['Sector']}, SNR={det['SNR']:.1f}, {det['N_transits']} eclipses"
            }
            vsx_ready.append(vsx_entry)
        
        if vsx_ready:
            pd.DataFrame(vsx_ready).to_csv(self.fm.files['vsx_ready'], index=False)
            print(f"\n  📝 VSX submission file ready: {self.fm.files['vsx_ready']}")

    def _report(self):
        """Generate comprehensive HTML report."""
        
        all_dets = [d['data'] for d in self.progress.data['detections'] if 'data' in d]
        
        # Count by type
        eb_count = sum(1 for d in all_dets if d.get('Classification', '').startswith('E'))
        planet_count = sum(1 for d in all_dets if 'planet' in d.get('Mode', '').lower())
        exotic_count = len(all_dets) - eb_count - planet_count
        multi_count = sum(1 for d in all_dets if d.get('Multi_Period', False))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Sector {self.sector} Enhanced Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f4f8; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ background: white; padding: 25px; border-radius: 10px; min-width: 140px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }}
        .stat:hover {{ transform: translateY(-5px); }}
        .stat-val {{ font-size: 32px; font-weight: bold; color: #667eea; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; }}
        th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .eb {{ background: #e3f2fd !important; }}
        .planet {{ background: #fff3e0 !important; }}
        .exotic {{ background: #f3e5f5 !important; }}
        .multi {{ background: #c8e6c9 !important; font-weight: bold; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔭 Sector {self.sector} Enhanced Discovery Report</h1>
        <p style="font-size: 16px; margin: 10px 0 0 0;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Modes: {', '.join(self.modes)}
        </p>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-val">{len(self.progress.data['processed'])}</div>
            <div class="stat-label">Stars Processed</div>
        </div>
        <div class="stat">
            <div class="stat-val">{len(all_dets)}</div>
            <div class="stat-label">Total Detections</div>
        </div>
        <div class="stat">
            <div class="stat-val">{eb_count}</div>
            <div class="stat-label">Eclipsing Binaries</div>
        </div>
        <div class="stat">
            <div class="stat-val">{planet_count}</div>
            <div class="stat-label">Planet Candidates</div>
        </div>
        <div class="stat">
            <div class="stat-val">{exotic_count}</div>
            <div class="stat-label">Exotic Systems</div>
        </div>
        <div class="stat">
            <div class="stat-val">{multi_count}</div>
            <div class="stat-label">Multi-Period Systems</div>
        </div>
    </div>

    <div class="section">
        <h2>🌟 All Detections</h2>
        <table>
            <tr>
                <th>TIC</th><th>Type</th><th>RA</th><th>Dec</th><th>Tmag</th>
                <th>Period</th><th>Depth%</th><th>SNR</th><th>Special</th><th>Plot</th>
            </tr>
"""

        for d in sorted(all_dets, key=lambda x: x.get('SNR', 0), reverse=True):
            row_class = ''
            if d.get('Multi_Period'):
                row_class = 'multi'
            elif 'planet' in d.get('Mode', '').lower():
                row_class = 'planet'
            elif d.get('Classification', '').startswith('E'):
                row_class = 'eb'
            else:
                row_class = 'exotic'
            
            special = []
            if d.get('Multi_Period'):
                special.append('🌟 Multi')
            if d.get('Planet_Prob', 0) > 0.7:
                special.append('🪐 High P')
            if d.get('Eccentricity', 0) > 0.5:
                special.append('💓 Heartbeat')
            special_str = ' '.join(special) if special else '-'
            
            html += f"""<tr class="{row_class}">
                <td><a href="https://exofop.ipac.caltech.edu/tess/target.php?id={d.get('TIC', '')}" target="_blank">
                    {d.get('TIC', '')}</a></td>
                <td>{d.get('Classification', 'Unknown')}</td>
                <td>{d.get('RA', 0):.4f}</td>
                <td>{d.get('Dec', 0):.4f}</td>
                <td>{d.get('Tmag', 0):.2f}</td>
                <td>{d.get('Period_d', 0):.6f}±{d.get('Period_err_d', 0):.6f}</td>
                <td>{d.get('Depth_pct', 0):.3f}</td>
                <td>{d.get('SNR', 0):.1f}</td>
                <td>{special_str}</td>
                <td><a href="plots/{d.get('Plot', '')}" target="_blank">View</a></td>
            </tr>
"""

        html += """
        </table>
    </div>
    
    <div class="section">
        <h3>📋 Data Files</h3>
        <ul>
            <li><a href="candidates_eb.csv">Eclipsing Binaries CSV</a></li>
            <li><a href="candidates_planet.csv">Planet Candidates CSV</a></li>
            <li><a href="candidates_exotic.csv">Exotic Systems CSV</a></li>
            <li><a href="candidates_all.csv">All Detections CSV</a></li>
            <li><a href="vsx_submissions.csv">VSX-Ready Submissions</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h3>Legend</h3>
        <p><span class="eb" style="padding: 5px;">Blue</span> = Eclipsing Binary</p>
        <p><span class="planet" style="padding: 5px;">Orange</span> = Planet Candidate</p>
        <p><span class="exotic" style="padding: 5px;">Purple</span> = Exotic System</p>
        <p><span class="multi" style="padding: 5px;">Green</span> = Multi-Period System</p>
    </div>
    
</body>
</html>
"""

        with open(self.fm.sector_dir / 'report.html', 'w') as f:
            f.write(html)

        print(f"\n  📄 Enhanced report: {self.fm.sector_dir}/report.html")

# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='DeepSkySurveyor v17 - Ultimate Multi-Mode Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection Modes:
  eb          - Standard eclipsing binaries
  planet      - Transiting exoplanets
  ultrashort  - Ultra-short period binaries (< 0.3 days)
  longperiod  - Long-period binaries (15-60 days)
  heartbeat   - Eccentric binaries with tidal distortion
  all         - All modes (default)

Examples:
  python DeepSkySurveyor_v17.py --mode all
  python DeepSkySurveyor_v17.py --mode eb planet --sector 5
  python DeepSkySurveyor_v17.py --mode exotic
        """
    )
    
    parser.add_argument('--sector', type=int, help='Specific sector to process')
    parser.add_argument('--mode', nargs='+', 
                       choices=['all', 'eb', 'planet', 'ultrashort', 'longperiod', 'heartbeat', 'exotic'],
                       default=['all'],
                       help='Detection modes to use')
    
    # Filter Jupyter/Colab kernel arguments
    filtered_args = [
        arg for arg in sys.argv[1:]
        if not arg.startswith('-f')
        and not arg.endswith('.json')
        and 'kernel' not in arg.lower()
    ]
    args = parser.parse_args(filtered_args if filtered_args else [])
    
    # Process mode selection
    if 'all' in args.mode:
        modes = ['eb', 'planet', 'ultrashort', 'longperiod', 'heartbeat']
    elif 'exotic' in args.mode:
        modes = ['ultrashort', 'longperiod', 'heartbeat']
    else:
        modes = args.mode
    
    try:
        EnhancedSurvey(sector=args.sector, modes=modes).run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress saved.\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
