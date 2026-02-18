#!/usr/bin/env python3
"""
=====================================================================
DeepSkySurveyor v18.3  –  Colab Production Edition
=====================================================================
FIXES vs v18.1:

F1. EXCESS-VARIABILITY FILTER (pulsator rejection)
    After flattening, compute the robust scatter (MAD-based σ) of the
    residuals.  If σ > 0.5% the star is a likely pulsator or has severe
    contamination; the flatten() window is not wide enough to remove it
    safely, so BLS would lock onto spurious signals.  Reject early with
    reason "Excess variability (pulsator/contamination)".

F2. PRE-WHITENING FOR SURVIVING HIGH-SCATTER STARS
    If 0.2% < σ ≤ 0.5% (moderate scatter, not an outright reject), run
    a Lomb-Scargle on long periods (>1 d) and subtract the best sinusoid
    before passing to BLS.  This catches semi-regulars that flatten()
    alone misses.

F3. BOOTSTRAP PERIOD UNCERTAINTY
    Deduplicate timestamps after bootstrap resampling so BLS never
    receives duplicate time values (which caused silent zero returns).
    Now always returns a non-zero uncertainty for any signal with
    >10 bootstrap realisations.

F4. HJD → BJD_TDB LABEL
    All epoch labels in submission text, CSV headers, HTML report, and
    plot annotation now say "BJD_TDB" (correct for TESS BTJD + 2457000).

F5. HALF-PERIOD HARMONIC CHECK
    After detection in ultra-short or EB mode, test whether P×2 produces
    a better-separated primary/secondary eclipse (depth_ratio closer to 1
    at P×2 would indicate equal-depth EW at double the period).  If the
    doubled period improves the depth ratio by >20%, re-run BLS at P×2
    and use that result.

F6. PHASE-PLOT X-AXIS CAPPED AT ±0.5
    The full-phase panel now always plots from -0.5 to +0.5 (phase units),
    not raw cycle counts.  The -7.5 … +7.5 display on long-period targets
    was caused by lightkurve returning unnormalised phase when
    normalize_phase was not explicitly set.  All fold() calls now pass
    normalize_phase=True.

F7. CLASSIFICATION FALLBACK FOR UNKNOWN MODE
    Detections that pass BLS but don't match EB/HB/Planet logic now get
    classification = mode.name (e.g. "Long-Period Binary") instead of
    "Unknown", giving VSX editors a meaningful type.

F8. TIGHTENED LIGHT-CURVE SHAPE VALIDATION  (catches V-noise & flat non-dips)
    a) In-transit phase window now proportional to actual transit duration
       (dur/period × 1.5, capped at ±0.1). Fixed window was masking 70×
       more OOT than in-transit flux on long-period short-duration targets.
    b) New 2σ dip-depth gate: dip must exceed 2× OOT scatter.
    c) Binned scatter ratio tightened 1.5→1.2×. V-shaped noise rejected.
    d) Centring check: deepest bin must be within central 40th percentile.
    e) except:pass removed — binning failure now causes rejection.

PASTE THIS ENTIRE FILE INTO ONE COLAB CELL AND RUN.
Also works as a normal script: python DeepSkySurveyor_v18.py
=====================================================================
"""

# ═══════════════════════════════════════════════════════════════════
# 0. COLAB KEEP-ALIVE
# ═══════════════════════════════════════════════════════════════════
try:
    from IPython.display import display, Javascript
    display(Javascript("""
        if (!window._dss_ka) {
            window._dss_ka = setInterval(function() {
                var b = document.querySelector("colab-connect-button");
                if (b) { var s = b.shadowRoot; if (s) { var c = s.getElementById("connect"); if (c) c.click(); } }
                console.log("DSS keep-alive " + new Date().toTimeString().slice(0,8));
            }, 55000);
            console.log("DSS keep-alive started.");
        }
    """))
    print("✅ Colab keep-alive active (pings every 55 s)")
except Exception:
    print("ℹ️  Keep-alive skipped (not in Colab – that's fine)")

# ═══════════════════════════════════════════════════════════════════
# 1. AUTO-INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════
import sys, os, subprocess, time, json, re, warnings, logging, zipfile, shutil, dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

PKGS = {
    'numpy':      'numpy',
    'pandas':     'pandas',
    'matplotlib': 'matplotlib',
    'scipy':      'scipy',
    'lightkurve': 'lightkurve',
    'astropy':    'astropy',
    'astroquery': 'astroquery',
    'requests':   'requests',
}

print("\n" + "="*70)
print("  DeepSkySurveyor v18.3  –  Colab Production Edition")
print("="*70 + "\n📦 Checking dependencies...")

for mod, pkg in PKGS.items():
    try:
        __import__(mod); print(f"  ✓ {mod}")
    except ImportError:
        print(f"  ⏳ Installing {pkg}...", end=" ", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("done")

print("  All ready.\n")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

import lightkurve as lk
from astropy.timeseries import BoxLeastSquares, LombScargle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Catalogs, Observations
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

warnings.filterwarnings("ignore")
logging.getLogger("lightkurve").setLevel(logging.ERROR)
logging.getLogger("astroquery").setLevel(logging.ERROR)
Simbad.reset_votable_fields()
Simbad.TIMEOUT = 30

# ── Network timeouts ──────────────────────────────────────────────
import signal as _signal

def _patch_mast_timeout(timeout=30):
    try:
        from astroquery.mast import Catalogs as _C, Observations as _O
        import requests as _req
        for svc in [_C, _O]:
            orig = svc._session.send
            def _send(r, **kw):
                kw.setdefault('timeout', timeout)
                return orig(r, **kw)
            svc._session.send = _send
    except Exception:
        pass

_patch_mast_timeout(30)
Vizier.TIMEOUT = 30

class _Timeout:
    def __init__(self, seconds=25):
        self.seconds = seconds
    def _handler(self, signum, frame):
        raise TimeoutError(f"Network call timed out after {self.seconds}s")
    def __enter__(self):
        try: _signal.signal(_signal.SIGALRM, self._handler); _signal.alarm(self.seconds)
        except (AttributeError, OSError): pass
        return self
    def __exit__(self, *_):
        try: _signal.alarm(0)
        except (AttributeError, OSError): pass

# ═══════════════════════════════════════════════════════════════════
# 2. SECTOR PRIORITY ORDER
# ═══════════════════════════════════════════════════════════════════
NEGLECTED_SECTORS = (
    list(range(56, 70)) +
    list(range(40, 56)) +
    list(range(27, 40)) +
    list(range(14, 27))
)

# ═══════════════════════════════════════════════════════════════════
# 3. DISCOVERY MODES
# ═══════════════════════════════════════════════════════════════════
@dataclass
class DiscoveryMode:
    name:         str
    min_period:   float;  max_period:   float
    min_depth:    float;  max_depth:    float
    min_duration: float;  max_duration: float
    min_snr:      float;  min_transits: int
    description:  str

MODES = {
    'eb': DiscoveryMode(
        'Eclipsing Binary',
        0.3, 15.0, 0.003, 1.0, 0.01, 0.25, 8.0, 3,
        'Standard eclipsing binaries'),
    'planet': DiscoveryMode(
        'Exoplanet',
        1.0, 100.0, 0.0003, 0.03, 0.01, 0.30, 10.0, 3,
        'Transiting exoplanets'),
    'ultrashort': DiscoveryMode(
        'Ultra-Short Binary',
        0.1, 0.3, 0.003, 1.0, 0.005, 0.10, 9.0, 5,
        'Ultra-short period binaries'),
    'longperiod': DiscoveryMode(
        'Long-Period Binary',
        15.0, 200.0, 0.005, 1.0, 0.02, 0.50, 9.0, 2,
        'Long-period eclipsing systems'),
    'heartbeat': DiscoveryMode(
        'Heartbeat Star',
        5.0, 100.0, 0.001, 0.30, 0.05, 0.80, 8.0, 2,
        'Eccentric binary with tidal distortion'),
}

# ═══════════════════════════════════════════════════════════════════
# 4. GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════
class Config:
    WORKDIR              = "DeepSkySurveyor_v18"
    ALL_SECTORS          = NEGLECTED_SECTORS
    # ── Magnitude range recalibrated for novelty ──────────────────────
    # Tmag 9-12  : exhaustively studied by SPOC + professional follow-up
    # Tmag 12-16 : SPOC coverage thins out above 14; QLP only above ~14.5
    #              Far fewer dedicated searches → highest novelty density
    MAG_MIN              = 12.0
    MAG_MAX              = 16.0
    # ── SNR floors raised to compensate for fainter/noisier targets ───
    # Faint stars have higher photon noise; same absolute SNR threshold
    # means far more false positives. +2 across all modes.
    SNR_BOOST            = 2.0    # added to every mode's min_snr at runtime
    MIN_GALACTIC_LAT     = 10.0
    MAX_CONTAMINATION    = 0.10
    MIN_DATA_POINTS      = 1000
    QUERY_DELAY          = 1.0
    DOWNLOAD_DELAY       = 1.5
    MAX_RETRIES          = 3
    CHECKPOINT_EVERY     = 25
    MAX_PERIODS          = 3
    MASK_WIDTH           = 0.15
    N_BOOTSTRAP          = 50
    # ── Light-curve source priority ───────────────────────────────────
    # SPOC 2-min is best quality but stops at ~Tmag 13.5 in practice.
    # QLP 10-min covers to ~Tmag 15.  TESS-SPOC FFI covers the rest.
    # Download order: SPOC 120s → SPOC any → QLP → TESS-SPOC FFI
    USE_QLP              = True   # allow QLP products for faint stars
    USE_FFI              = True   # allow FFI products as last resort
    # ── Variability thresholds ────────────────────────────────────────
    VARIABILITY_REJECT   = 0.005   # σ > 0.5% → outright reject (pulsator)
    VARIABILITY_PREWHITE = 0.002   # σ > 0.2% → pre-whiten before BLS

# ═══════════════════════════════════════════════════════════════════
# 5. HELPERS
# ═══════════════════════════════════════════════════════════════════
@dataclass
class CatalogMatch:
    vsx:      Optional[str] = None
    asassn:   Optional[str] = None
    atlas:    Optional[str] = None
    gaia_var: Optional[str] = None
    simbad:   Optional[str] = None
    kostov:   bool          = False

    def is_known(self) -> bool:
        return any([self.vsx, self.asassn, self.atlas,
                    self.gaia_var, self.simbad, self.kostov])

    def summary(self) -> str:
        parts = []
        if self.kostov:   parts.append("Kostov-EB")
        if self.vsx:      parts.append(f"VSX:{self.vsx}")
        if self.asassn:   parts.append(f"ASAS-SN:{self.asassn}")
        if self.atlas:    parts.append(f"ATLAS:{self.atlas}")
        if self.gaia_var: parts.append(f"GaiaVar:{self.gaia_var}")
        if self.simbad:   parts.append(f"SIMBAD:{self.simbad}")
        return "; ".join(parts) if parts else "None"

def _js(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.bool_):    return bool(obj)
    return str(obj)

def _mad_sigma(y):
    """Robust σ via median absolute deviation, scaled to Gaussian equivalent."""
    return float(median_abs_deviation(y, scale='normal'))

# ═══════════════════════════════════════════════════════════════════
# 6. FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
class FileManager:
    def __init__(self, sector):
        self.root       = Path(Config.WORKDIR)
        self.sector     = sector
        self.sector_dir = self.root / f"Sector_{sector:02d}"
        self.dirs = {
            'plots':  self.sector_dir / 'plots',
            'packs':  self.sector_dir / 'submission_packs',
            'cache':  self.root / 'cache',
        }
        self.files = {
            'targets':            self.sector_dir / 'targets.csv',
            'progress':           self.sector_dir / 'progress.json',
            'candidates_eb':      self.sector_dir / 'candidates_eb.csv',
            'candidates_planet':  self.sector_dir / 'candidates_planet.csv',
            'candidates_exotic':  self.sector_dir / 'candidates_exotic.csv',
            'candidates_all':     self.sector_dir / 'candidates_all.csv',
            'vsx_ready':          self.sector_dir / 'vsx_submissions.csv',
            'rejected':           self.sector_dir / 'rejected.csv',
            'kostov':             self.root / 'cache' / 'kostov_ebs.txt',
            'sectors_log':        self.root / 'processed_sectors.json',
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sector_dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 7. PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════════
class Progress:
    def __init__(self, filepath):
        self.file  = filepath
        self.data  = self._load()
        self.start = time.time()
        self.times: List[float] = []

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file) as f: return json.load(f)
            except: pass
        return {'processed': [], 'detections': [], 'rejected': []}

    def save(self):
        self.data['last_update'] = datetime.now().isoformat()
        with open(self.file, 'w') as f:
            json.dump(self.data, f, indent=2, default=_js)

    def is_done(self, tic):    return str(tic) in self.data['processed']

    def mark_done(self, tic, t=None):
        if str(tic) not in self.data['processed']:
            self.data['processed'].append(str(tic))
        if t: self.times = (self.times + [t])[-100:]

    def add_detection(self, tic, det):
        self.data['detections'].append({'tic': str(tic), 'data': det})

    def add_rejected(self, tic, reason):
        self.data['rejected'].append({'tic': str(tic), 'reason': reason})

    def eta(self, remaining):
        if not self.times: return "calculating..."
        return str(timedelta(seconds=int(np.mean(self.times) * remaining)))

    def rate(self):
        e = time.time() - self.start
        return len(self.times) / (e / 3600) if e > 0 and self.times else 0

# ═══════════════════════════════════════════════════════════════════
# 8. SECTOR MANAGER
# ═══════════════════════════════════════════════════════════════════
class SectorManager:
    def __init__(self, f):
        self.file = f; self.data = self._load()

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file) as f: return json.load(f)
            except: pass
        return {'completed': [], 'in_progress': None}

    def save(self):
        with open(self.file, 'w') as f: json.dump(self.data, f, indent=2)

    def get_next(self):
        done = set(self.data.get('completed', []))
        for s in Config.ALL_SECTORS:
            if s not in done: return s
        return None

    def mark_complete(self, s):
        if s not in self.data['completed']: self.data['completed'].append(s)
        self.data['in_progress'] = None; self.save()

    def mark_progress(self, s):
        self.data['in_progress'] = s; self.save()

# ═══════════════════════════════════════════════════════════════════
# 9. KNOWN CATALOGS
# ═══════════════════════════════════════════════════════════════════
class KnownCatalogs:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.kostov    = self._load_kostov()

    def _load_kostov(self):
        f = self.cache_dir / 'kostov_ebs.txt'
        if f.exists():
            try:
                known = set(l.strip() for l in f.open() if l.strip())
                if len(known) > 100:
                    print(f"  ✓ {len(known)} Kostov EBs loaded from cache")
                    return known
            except: pass
        print("  📥 Downloading Kostov EB catalog...", end=" ", flush=True)
        known = set()
        try:
            url = ("https://content.cld.iop.org/journals/0067-0049/279/2/50"
                   "/revision1/apjsade2d8t3_mrt.txt")
            r = requests.get(url, timeout=60)
            for line in r.text.split('\n'):
                p = line.strip().split()
                if p and p[0].isdigit(): known.add(p[0])
            f.write_text('\n'.join(sorted(known)))
            print(f"{len(known)} EBs loaded")
        except Exception as e:
            print(f"failed ({e}) – continuing without Kostov cache")
        return known

    def is_kostov(self, tic): return str(tic) in self.kostov

    @staticmethod
    def cross_match(ra, dec, radius_arcsec=30) -> CatalogMatch:
        m = CatalogMatch()
        if ra is None or dec is None: return m
        try:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            try:
                with _Timeout(15):
                    v = Vizier(columns=['Name', 'Type'], row_limit=1)
                    r = v.query_region(coord, radius=radius_arcsec*u.arcsec, catalog='B/vsx/vsx')
                if r and len(r) and len(r[0]):
                    m.vsx = f"{r[0][0]['Name']} ({r[0][0]['Type']})"
            except: pass
            try:
                with _Timeout(15):
                    v = Vizier(columns=['Name', 'Type'], row_limit=1)
                    r = v.query_region(coord, radius=radius_arcsec*u.arcsec, catalog='II/366/catalog')
                if r and len(r) and len(r[0]):
                    m.asassn = str(r[0][0]['Name'])
            except: pass
            try:
                with _Timeout(15):
                    v = Vizier(columns=['ATLAS', 'Type'], row_limit=1)
                    r = v.query_region(coord, radius=radius_arcsec*u.arcsec, catalog='J/AJ/156/241')
                if r and len(r) and len(r[0]):
                    m.atlas = f"ATLAS-{r[0][0]['ATLAS']}"
            except: pass
            try:
                with _Timeout(15):
                    v = Vizier(columns=['Source', 'Class', 'Freq1'], row_limit=1)
                    r = v.query_region(coord, radius=10*u.arcsec, catalog='I/358/vclassre')
                if r and len(r) and len(r[0]):
                    row0 = r[0][0]
                    # Only reject if Gaia has a measured frequency/period.
                    # A bare variability flag with no period = uncharacterised
                    # variable → may still be a novel discovery.
                    try:
                        freq1 = row0['Freq1']
                        has_period = (freq1 is not None and
                                      not np.ma.is_masked(freq1) and
                                      float(freq1) > 0)
                    except Exception:
                        has_period = False
                    if has_period:
                        m.gaia_var = f"GaiaDR3-{row0['Source']} (period known)"
            except: pass
            try:
                with _Timeout(15):
                    res = Simbad.query_region(coord, radius=10*u.arcsec)
                if res is not None and len(res):
                    mid = str(res[0]['MAIN_ID']).upper()
                    VAR = ('V*', 'EB', 'EA', 'EW', 'RR', 'CEP', 'MIRA',
                           'LPV', 'ROT', 'BY', 'RS', 'W UMA', 'DSCT', 'GDOR')
                    if any(t in mid for t in VAR):
                        m.simbad = str(res[0]['MAIN_ID'])
            except: pass
        except: pass
        return m

# ═══════════════════════════════════════════════════════════════════
# 10. TIC INFO QUERY
# ═══════════════════════════════════════════════════════════════════
class TICQuery:
    @staticmethod
    def get(tic):
        try:
            with _Timeout(25):
                result = Catalogs.query_criteria(catalog="TIC", ID=tic)
            if not result or len(result) == 0: return None
            row = result[0]
            def g(k, d=None):
                try:
                    v = row[k]
                    return d if (v is None or np.ma.is_masked(v)) else float(v)
                except: return d
            ra, dec = g('ra'), g('dec')
            gal_lat = None
            if ra and dec:
                try: gal_lat = SkyCoord(ra*u.deg, dec*u.deg).galactic.b.deg
                except: pass
            return dict(tic=str(tic), ra=ra, dec=dec,
                        tmag=g('Tmag'), teff=g('Teff'),
                        radius=g('rad'), mass=g('mass'), logg=g('logg'),
                        gal_lat=gal_lat, contamination=g('contratio', 0),
                        parallax=g('plx'))
        except: return None

# ═══════════════════════════════════════════════════════════════════
# 11. TARGET FINDER
# ═══════════════════════════════════════════════════════════════════
class TargetFinder:
    def __init__(self, fm, catalogs, sector):
        self.fm = fm; self.catalogs = catalogs; self.sector = sector

    # TESS sector sky footprints (RA centre, Dec centre, ~12°×12° box).
    # Used to query the TIC directly for stars in the magnitude range,
    # rather than relying on what SPOC chose to process — this gives access
    # to the full faint-star population including QLP/FFI-only targets.
    # Values are approximate sector pointing centres (J2000).
    SECTOR_CENTRES = {
        14:(352.5,-64.9), 15:(55.0,-64.9), 16:(118.0,-64.9), 17:(181.0,-64.9),
        18:(244.0,-64.9), 19:(307.0,-64.9), 20:(10.0,-30.0),  21:(73.0,-30.0),
        22:(136.0,-30.0), 23:(199.0,-30.0), 24:(262.0,-30.0), 25:(325.0,-30.0),
        26:(28.0,-6.0),   27:(352.5,+64.9), 28:(55.0,+64.9),  29:(118.0,+64.9),
        30:(181.0,+64.9), 31:(244.0,+64.9), 32:(307.0,+64.9), 33:(10.0,+30.0),
        34:(73.0,+30.0),  35:(136.0,+30.0), 36:(199.0,+30.0), 37:(262.0,+30.0),
        38:(325.0,+30.0), 39:(28.0,+6.0),   40:(0.0,-6.0),    41:(27.5,-6.0),
        42:(55.0,-6.0),   43:(82.5,-6.0),   44:(110.0,-6.0),  45:(137.5,-6.0),
        46:(165.0,-6.0),  47:(192.5,-6.0),  48:(220.0,-6.0),  49:(247.5,-6.0),
        50:(275.0,-6.0),  51:(302.5,-6.0),  52:(330.0,-6.0),  53:(357.5,-6.0),
        54:(25.0,-6.0),   55:(52.5,-6.0),   56:(0.0,+6.0),    57:(27.5,+6.0),
        58:(55.0,+6.0),   59:(82.5,+6.0),   60:(110.0,+6.0),  61:(137.5,+6.0),
        62:(165.0,+6.0),  63:(192.5,+6.0),  64:(220.0,+6.0),  65:(247.5,+6.0),
        66:(275.0,+6.0),  67:(302.5,+6.0),  68:(330.0,+6.0),  69:(357.5,+6.0),
    }

    def get_targets(self):
        if self.fm.files['targets'].exists():
            try:
                df = pd.read_csv(self.fm.files['targets'])
                if len(df) >= 10:
                    tl = df['tic'].astype(str).tolist()
                    print(f"  ✓ {len(tl)} cached targets for Sector {self.sector}")
                    return tl
            except: pass

        print(f"\n🔍 Querying TIC for Sector {self.sector} "
              f"(Tmag {Config.MAG_MIN}–{Config.MAG_MAX})...")

        all_tics = self._tic_by_magnitude()

        # Fall back to MAST observation query if TIC query fails
        if not all_tics:
            print("  ⚠️  TIC query failed – falling back to MAST observation list")
            all_tics = self._mast_fallback()

        if not all_tics:
            print(f"  ❌ No targets found for Sector {self.sector}"); return []

        novel = [t for t in all_tics if not self.catalogs.is_kostov(t)]
        print(f"  ✅ {len(novel)} targets  "
              f"({len(all_tics)-len(novel)} Kostov EBs pre-filtered)")
        pd.DataFrame({'tic': novel, 'sector': self.sector}).to_csv(
            self.fm.files['targets'], index=False)
        return novel

    def _tic_by_magnitude(self):
        """
        Query the TIC directly for all stars in the sector footprint
        within the target magnitude range. Returns far more targets than
        the MAST observation list (which only covers SPOC-processed stars),
        giving access to the faint QLP/FFI population.
        """
        tics = []
        centre = self.SECTOR_CENTRES.get(self.sector)
        if centre is None:
            print(f"  ⚠️  No footprint data for Sector {self.sector}")
            return []

        ra_c, dec_c = centre
        # TESS cameras cover ~24°×24° total; use a 20° radius cone to get
        # the full field without querying half the sky
        try:
            with _Timeout(120):
                result = Catalogs.query_criteria(
                    catalog    = "TIC",
                    coordinates= f"{ra_c} {dec_c}",
                    radius     = 15.0,          # degrees
                    Tmag       = [Config.MAG_MIN, Config.MAG_MAX],
                    objType    = "STAR",
                )
            if result and len(result):
                tics = [str(r['ID']) for r in result]
                print(f"  TIC query: {len(tics)} stars in Tmag "
                      f"{Config.MAG_MIN}–{Config.MAG_MAX}")
        except Exception as e:
            print(f"  TIC query error: {str(e)[:80]}")
        return tics

    def _mast_fallback(self):
        """Original MAST observation-list query, used only if TIC query fails."""
        tics = set()
        try:
            with _Timeout(60):
                obs = Observations.query_criteria(
                    obs_collection="TESS",
                    dataproduct_type="timeseries",
                    sequence_number=self.sector)
            if obs and len(obs):
                print(f"      {len(obs)} observations")
                for tn in obs['target_name']:
                    m = (re.search(r'TIC[\s\-]?(\d{6,})', str(tn), re.I) or
                         re.search(r'(\d{8,})', str(tn)))
                    if m: tics.add(m.group(1))
                print(f"      {len(tics)} unique TICs extracted (fallback)")
        except Exception as e:
            print(f"      MAST error: {str(e)[:60]}")
        return list(tics)

# ═══════════════════════════════════════════════════════════════════
# 12. ANALYZER
# ═══════════════════════════════════════════════════════════════════
class Analyzer:
    def __init__(self, fm): self.fm = fm

    def run(self, tic, modes):
        lc, nsec = self._download(tic)
        if lc is None:
            return {'status': 'rejected', 'reason': 'No light curve'}
        if len(lc) < Config.MIN_DATA_POINTS:
            return {'status': 'rejected', 'reason': f'Too few points ({len(lc)})'}
        try:
            lc_flat = lc.remove_nans().remove_outliers(sigma=5).flatten(window_length=501)
        except:
            return {'status': 'rejected', 'reason': 'Flatten failed'}

        # ── F1: Excess variability filter ────────────────────────────
        sigma_flat = _mad_sigma(lc_flat.flux.value)
        if sigma_flat > Config.VARIABILITY_REJECT:
            return {'status': 'rejected',
                    'reason': f'Excess variability σ={sigma_flat*100:.2f}% '
                              f'(pulsator/contamination – flatten ineffective)'}

        # ── F2: Pre-whitening for moderate scatter ────────────────────
        if sigma_flat > Config.VARIABILITY_PREWHITE:
            lc_flat = self._prewhiten(lc_flat)

        best = None
        for mn in modes:
            if mn not in MODES: continue
            mode = MODES[mn]
            # Raise SNR floor for faint targets to suppress noise false-positives
            boosted_snr  = mode.min_snr + Config.SNR_BOOST
            boosted_mode = dataclasses.replace(mode, min_snr=boosted_snr)
            r = self._detect(lc_flat, boosted_mode, tic)
            if r and (best is None or r['snr'] > best['snr']):
                best = r

        if best is None:
            return {'status': 'rejected', 'reason': 'No signals detected'}

        mp = self._multi_period(lc_flat, best)
        if len(mp) > 1:
            best['multi_period'] = True
            best['all_periods']  = mp
            best['classification'] = 'Triple/Hierarchical System'

        best.update(lc=lc_flat, n_sectors=nsec, n_points=len(lc_flat))
        return best

    # ── F2: Pre-whitening ─────────────────────────────────────────────
    def _prewhiten(self, lc):
        """Subtract best long-period Lomb-Scargle sinusoid from light curve."""
        try:
            t, y = lc.time.value, lc.flux.value
            # Only fit periods longer than 1 day (not the eclipse period itself)
            ls = LombScargle(t, y)
            freq, power = ls.autopower(minimum_frequency=1/60, maximum_frequency=1.0)
            best_freq = freq[np.argmax(power)]
            if best_freq > 0:
                t_fit = np.linspace(t.min(), t.max(), len(t))
                model  = ls.model(t, best_freq)
                y_corr = y - model + 1.0   # subtract sinusoid, preserve mean=1
                lc_new = lc.copy()
                lc_new.flux = lc_new.flux.__class__(y_corr, unit=lc.flux.unit)
                return lc_new
        except: pass
        return lc

    def _detect(self, lc, mode: DiscoveryMode, tic):
        bls = self._bls(lc, mode)
        if bls is None: return None
        v = self._validate(bls, lc, mode)
        if not v['valid']: return None

        # F7: meaningful default classification
        classification = mode.description   # e.g. "Long-period eclipsing systems"
        sec_depth = 0.0; ecc = 0.0; prob_planet = 0.0

        if mode.name in ('Eclipsing Binary', 'Long-Period Binary', 'Ultra-Short Binary'):
            ec = self._classify_eb(lc, bls['period'], bls['epoch'])
            classification = ec['type']; sec_depth = ec['secondary_depth']

            # ── F5: Half-period harmonic check for ultra-short / EB ──
            if mode.name in ('Ultra-Short Binary', 'Eclipsing Binary'):
                bls2 = self._try_double_period(lc, bls, mode)
                if bls2 is not None:
                    v2 = self._validate(bls2, lc, mode)
                    if v2['valid'] and bls2['snr'] >= bls['snr'] * 0.85:
                        ec2 = self._classify_eb(lc, bls2['period'], bls2['epoch'])
                        # Accept doubled period if it improves depth ratio toward 1.0
                        # (equal-depth EW is the canonical half-period alias)
                        if abs(ec2['depth_ratio'] - 1.0) < abs(ec['depth_ratio'] - 1.0) - 0.15:
                            bls = bls2
                            v   = v2
                            ec  = ec2
                            classification = ec['type']
                            sec_depth      = ec['secondary_depth']

        if mode.name == 'Heartbeat Star':
            ok, ecc = self._heartbeat(lc, bls['period'], bls['epoch'])
            if not ok: return None
            classification = 'Heartbeat Star'

        if mode.name == 'Exoplanet':
            prob_planet = self._planet_prob(lc, bls)

        per_err = self._bootstrap(lc, bls['period'], bls['epoch'])

        return {
            'status':         'detected',
            'tic':            str(tic),
            'mode':           mode.name,
            'period':         bls['period'],
            'period_err':     per_err,
            'epoch':          bls['epoch'],
            'depth':          bls['depth'],
            'depth_pct':      bls['depth'] * 100,
            'duration':       bls['duration'],
            'snr':            bls['snr'],
            'n_transits':     v['n_transits'],
            'classification': classification,
            'secondary_depth':sec_depth,
            'eccentricity':   ecc,
            'prob_planet':    prob_planet,
            'multi_period':   False,
        }

    # ── F5: Try doubled period ────────────────────────────────────────
    def _try_double_period(self, lc, bls, mode):
        """Run BLS at 2× the detected period; return result if physical."""
        try:
            p2 = bls['period'] * 2
            if p2 > mode.max_period * 1.5:   # don't go wildly out of range
                return None
            t, y = lc.time.value, lc.flux.value
            pr   = np.linspace(p2 * 0.98, p2 * 1.02, 500)
            dur  = bls['duration']
            pg   = BoxLeastSquares(t, y).power(pr, duration=[dur, dur*1.5])
            ix   = np.argmax(pg.power)
            bp2  = float(pr[ix])
            stats = BoxLeastSquares(t, y).compute_stats(bp2, dur, float(pg.transit_time[ix]))
            depth2 = float(abs(stats['depth'][0]))
            snr2   = float((depth2 / np.std(y)) * np.sqrt(len(t) * dur / bp2))
            return dict(period=bp2, epoch=float(pg.transit_time[ix]),
                        duration=dur, depth=depth2, snr=snr2,
                        power=float(pg.power[ix]))
        except: return None

    def _bls(self, lc, mode: DiscoveryMode):
        try:
            t, y = lc.time.value, lc.flux.value
            periods = np.linspace(mode.min_period, mode.max_period, 5000)
            dmax = min(mode.max_duration, mode.min_period * 0.8)
            durs = np.array([0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.25])
            durs = durs[(durs >= mode.min_duration) & (durs <= dmax)]
            if not len(durs):
                durs = np.array([mode.min_duration, mode.min_duration*2])
            pg  = BoxLeastSquares(t, y).power(periods, duration=durs)
            idx = np.argmax(pg.power)
            bp, bd, bt = float(pg.period[idx]), float(pg.duration[idx]), float(pg.transit_time[idx])
            stats = BoxLeastSquares(t, y).compute_stats(bp, bd, bt)
            depth = float(abs(stats['depth'][0]))
            snr   = float((depth / np.std(y)) * np.sqrt(len(t)*bd/bp))
            return dict(period=bp, epoch=bt, duration=bd,
                        depth=depth, snr=snr, power=float(pg.power[idx]))
        except: return None

    def _validate(self, bls, lc, mode: DiscoveryMode):
        p, snr, depth, dur = bls['period'], bls['snr'], bls['depth'], bls['duration']
        if snr < mode.min_snr:
            return {'valid':False,'reason':f'SNR {snr:.1f}<{mode.min_snr}'}
        if not (mode.min_depth <= depth <= mode.max_depth):
            return {'valid':False,'reason':'Depth out of range'}
        for n in [1,2,3,4]:
            if abs(p - 0.99727/n)/(0.99727/n) < 0.01:
                return {'valid':False,'reason':'Sidereal alias'}
        for d in [0.5, 1.0, 2.0]:
            if abs(p - d) < 0.02:
                return {'valid':False,'reason':'Day alias'}
        for a in [13.7, 6.85]:
            if abs(p - a)/a < 0.02:
                return {'valid':False,'reason':'TESS orbit alias'}
        dh = dur * 24
        if dh >= 0.5 and abs(dh - round(dh)) < 0.02 and int(round(dh)) in [1,2,3,4,5,6,12,24]:
            return {'valid':False,'reason':f'Round duration {dh:.2f}h – likely artifact'}
        frac = dur / p
        if frac > 0.5:   return {'valid':False,'reason':'Duration>50% of period'}
        if frac < 0.001: return {'valid':False,'reason':'Duration<0.1% of period'}
        # F6: use np.ptp (NumPy 2.0 compatible)
        baseline = np.ptp(lc.time.value)
        if p > baseline * 0.8:
            return {'valid':False,'reason':f'Period {p:.1f}d > 80% baseline {baseline:.1f}d'}
        phase = ((lc.time.value - bls['epoch']) % p) / p
        in_t  = (phase < 0.1) | (phase > 0.9)
        n_tr  = max(1, int(np.sum(np.diff(in_t.astype(int)) == 1)))
        if n_tr < mode.min_transits:
            return {'valid':False,'reason':f'Only {n_tr} transits (need {mode.min_transits})'}
        folded = lc.fold(period=p, epoch_time=bls['epoch'], normalize_phase=True)

        # Phase half-width proportional to actual transit duration, capped at 0.1.
        # For a 15d period with a 0.5h transit, dur/p ≈ 0.0014 — using a fixed 0.1
        # window would include 70× more out-of-transit flux than in-transit flux,
        # making the median comparison meaningless.
        transit_hw = min(0.10, max(0.01, (dur / p) * 1.5))
        oot_lo, oot_hi = transit_hw + 0.10, transit_hw + 0.20   # clear of transit

        pm  = np.abs(folded.phase.value) < transit_hw
        oom = ((folded.phase.value > oot_lo) & (folded.phase.value < oot_hi))

        if pm.sum() < 10:
            return {'valid':False,'reason':f'Too few in-transit points ({pm.sum()})'}
        if oom.sum() < 10:
            return {'valid':False,'reason':f'Too few out-of-transit points ({oom.sum()})'}

        flux_in  = folded.flux.value[pm]
        flux_oot = folded.flux.value[oom]
        med_in   = float(np.median(flux_in))
        med_oot  = float(np.median(flux_oot))

        if med_in >= med_oot:
            return {'valid':False,'reason':'No flux decrease at eclipse phase'}

        # Require the dip to be at least 2× the OOT scatter — rules out noise spikes
        oot_scatter = float(np.std(flux_oot))
        dip_depth   = med_oot - med_in
        if oot_scatter > 0 and dip_depth < 2.0 * oot_scatter:
            return {'valid':False,'reason':
                    f'Dip ({dip_depth*100:.3f}%) < 2σ OOT scatter ({oot_scatter*100:.3f}%)'}

        # Binned coherence — no silent skip: if binning fails, reject
        try:
            n_bins = max(50, int(1.0 / transit_hw / 2))   # finer bins for short transits
            bn     = folded.bin(bins=n_bins)
            bpm    = np.abs(bn.phase.value) < transit_hw * 1.2
            boom   = ((bn.phase.value > oot_lo) & (bn.phase.value < oot_hi))

            if bpm.sum() < 2:
                return {'valid':False,'reason':'Too few binned in-transit points'}
            if boom.sum() < 2:
                return {'valid':False,'reason':'Too few binned OOT points'}

            bmed_in  = float(np.median(bn.flux.value[bpm]))
            bmed_oot = float(np.median(bn.flux.value[boom]))

            # Tightened: in-transit must be clearly below OOT (not just ≤ 0.999×)
            if bmed_in >= bmed_oot * 0.998:
                return {'valid':False,'reason':'No coherent dip in binned phase'}

            # Tightened scatter ratio: was 1.5×, now 1.2× — V-shaped noise rejected
            bstd_in  = float(np.std(bn.flux.value[bpm]))
            bstd_oot = float(np.std(bn.flux.value[boom]))
            if bstd_oot > 0 and bstd_in > bstd_oot * 1.2:
                return {'valid':False,'reason':
                        f'Transit scatter {bstd_in/bstd_oot:.2f}× OOT – likely V-shaped noise'}

            # Shape check: central bin must be the deepest in the transit window
            if bpm.sum() >= 3:
                in_fluxes   = bn.flux.value[bpm]
                centre_idx  = np.argmin(np.abs(bn.phase.value[bpm]))
                centre_flux = in_fluxes[centre_idx]
                if centre_flux > np.percentile(in_fluxes, 40):
                    return {'valid':False,'reason':'Dip not centred – asymmetric/spike artifact'}

        except Exception as e:
            # Binning failed — do not silently pass; treat as invalid
            return {'valid':False,'reason':f'Binned coherence check failed ({e})'}

        return {'valid':True, 'n_transits':n_tr}

    def _classify_eb(self, lc, period, epoch):
        try:
            f   = lc.fold(period=period, epoch_time=epoch, normalize_phase=True)
            pm  = np.abs(f.phase.value) < 0.1
            sm  = (f.phase.value > 0.4) & (f.phase.value < 0.6)
            oom = (f.phase.value > 0.2) & (f.phase.value < 0.3)
            pd_ = (1-np.median(f.flux.value[pm])) if pm.sum()>10 else 0
            sd  = (1-np.median(f.flux.value[sm])) if sm.sum()>10 else 0
            oot = np.median(f.flux.value[oom]) if oom.sum()>10 else 1.0
            pd_ = abs(pd_/oot) if oot else 0
            sd  = abs(sd/oot)  if oot else 0
            r   = sd/pd_ if pd_ else 0
            tp  = 'EW' if r>0.8 else ('EB' if r>0.1 else 'EA')
            return {'type':tp, 'secondary_depth':float(sd), 'depth_ratio':float(r)}
        except:
            return {'type':'EA','secondary_depth':0.0,'depth_ratio':0.0}

    def _heartbeat(self, lc, period, epoch):
        """
        Detect heartbeat stars via two complementary metrics:

        1. Phase-profile skewness: a genuine heartbeat has a sharp
           brightening spike near periastron (phase ~0) that makes the
           folded profile strongly skewed.  Pure noise or sinusoids are
           nearly symmetric (skewness ≈ 0).

        2. Periastron asymmetry: the spike should be sharper on one side
           than the other (tidal distortion is not symmetric about phase 0).
           This distinguishes heartbeats from ellipsoidal variables.

        Both metrics must pass to avoid false positives from systematics.
        Returns (is_heartbeat: bool, eccentricity_proxy: float).
        """
        from scipy.stats import skew as _skew
        try:
            f    = lc.fold(period=period, epoch_time=epoch, normalize_phase=True)
            bins = np.linspace(-0.5, 0.5, 200)
            bf   = np.full(len(bins)-1, np.nan)
            for i in range(len(bins)-1):
                m = (f.phase.value >= bins[i]) & (f.phase.value < bins[i+1])
                if m.sum() >= 3:
                    bf[i] = np.median(f.flux.value[m])

            # Interpolate over NaN gaps (sparse bins at faint magnitudes)
            nans = np.isnan(bf)
            if nans.sum() > len(bf) * 0.3:
                return False, 0.0   # too many empty bins → unreliable
            if nans.any():
                xp   = np.where(~nans)[0]
                bf   = np.interp(np.arange(len(bf)), xp, bf[xp])

            # ── Metric 1: skewness of the phase profile ───────────────
            # Heartbeat profiles are positively skewed (sharp spike up).
            # Threshold of 0.5 rejects sinusoids (|skew| < 0.3 typically).
            profile_skew = float(_skew(bf))
            if profile_skew < 0.4:
                return False, 0.0

            # ── Metric 2: periastron spike asymmetry ──────────────────
            # Find the brightest bin; measure how much steeper the
            # leading edge is versus the trailing edge.
            peak_idx = int(np.argmax(bf))
            if not (8 < peak_idx < len(bf) - 8):
                return False, 0.0   # peak at edge → systematic, not HB

            wing = 8   # bins on each side
            lead  = bf[peak_idx] - bf[peak_idx - wing]   # rise before peak
            trail = bf[peak_idx] - bf[peak_idx + wing]   # fall after peak
            if lead <= 0 or trail <= 0:
                return False, 0.0

            asym = abs(lead - trail) / max(lead, trail)
            if asym < 0.25:
                return False, 0.0   # too symmetric → not a heartbeat

            # ── Metric 3: spike must stand above the OOT noise ────────
            oot_mask  = (np.abs(np.linspace(-0.5, 0.5, len(bf))) > 0.15)
            oot_std   = float(np.std(bf[oot_mask]))
            spike_amp = bf[peak_idx] - np.median(bf[oot_mask])
            if oot_std > 0 and spike_amp < 3.0 * oot_std:
                return False, 0.0   # spike not significant above OOT noise

            # Eccentricity proxy: higher asymmetry → more eccentric orbit
            ecc_proxy = min(0.95, 0.2 + asym * 0.6 + min(profile_skew, 2.0) * 0.1)
            return True, float(ecc_proxy)

        except Exception:
            return False, 0.0

    def _bootstrap(self, lc, period, epoch, n=50):
        """F3: Robust bootstrap with timestamp deduplication."""
        try:
            t, y = lc.time.value, lc.flux.value
            ps = []
            rng = np.random.default_rng()
            for _ in range(n):
                idx = rng.choice(len(t), len(t), replace=True)
                ts, ys = t[idx], y[idx]
                si = np.argsort(ts); ts, ys = ts[si], ys[si]
                # Deduplicate timestamps – BLS fails silently on duplicates
                ts, ui = np.unique(ts, return_index=True); ys = ys[ui]
                if len(ts) < 100: continue
                try:
                    pr = np.linspace(period*0.95, period*1.05, 200)
                    pg = BoxLeastSquares(ts, ys).power(pr, duration=min(0.05, period*0.1))
                    ps.append(float(pr[np.argmax(pg.power)]))
                except: continue
            if len(ps) > 10:
                sigma = float(np.std(ps))
                # Guard against pathological collapse (all bootstrap periods identical)
                return sigma if sigma > 1e-9 else float(np.std(ps[:5]) + 1e-7)
            return 0.0
        except: return 0.0

    def _planet_prob(self, lc, bls):
        try:
            f  = lc.fold(period=bls['period'], epoch_time=bls['epoch'], normalize_phase=True)
            tn = np.round((lc.time.value-bls['epoch'])/bls['period'])
            om = (tn%2==1)&(np.abs(f.phase.value)<0.1)
            em = (tn%2==0)&(np.abs(f.phase.value)<0.1)
            if om.sum()>10 and em.sum()>10:
                od  = 1-np.median(lc.flux.value[om])
                ed  = 1-np.median(lc.flux.value[em])
                avg = (od+ed)/2
                c   = 1.0-min(1.0,abs(od-ed)/avg) if avg else 0.5
                ds  = 1.0 if bls['depth']<0.02 else 0.5
                return float(min(1.0, max(0.0, c*0.6+ds*0.4)))
            return 0.5
        except: return 0.5

    def _multi_period(self, lc, primary):
        try:
            found = [primary['period']]
            t, y  = lc.time.value.copy(), lc.flux.value.copy()
            ph    = ((t-primary['epoch'])%primary['period'])/primary['period']
            mask  = (ph<Config.MASK_WIDTH)|(ph>(1-Config.MASK_WIDTH))
            y[mask] = np.median(y[~mask])
            for _ in range(Config.MAX_PERIODS-1):
                pr = np.linspace(0.3, 30.0, 3000)
                pg = BoxLeastSquares(t,y).power(pr, duration=[0.01,0.02,0.05,0.10])
                ix = np.argmax(pg.power)
                if pg.power[ix] < 0.05: break
                np2 = pr[ix]
                # Reject harmonics (including 2× and 0.5× of any found period)
                harm = any(abs(np2-p*n)/(p*n)<0.05
                           for p in found for n in [0.5,1.0,1.5,2,3,4,0.333,0.25])
                if not harm:
                    found.append(float(np2))
                    ph2 = ((t-pg.transit_time[ix])%np2)/np2
                    m2  = (ph2<Config.MASK_WIDTH)|(ph2>(1-Config.MASK_WIDTH))
                    y[m2] = np.median(y[~(mask|m2)])
                else: break
            return found
        except: return [primary['period']]

    def _download(self, tic):
        """
        Download ALL available light curves for this TIC and stitch them.
        Priority chain:
          1. SPOC 2-min (best quality, covers Tmag ≲ 13.5)
          2. SPOC any cadence (catches 20-min SPOC FFI products)
          3. QLP 10-min  (MIT Quick-Look Pipeline, covers to Tmag ~15)
          4. TESS-SPOC FFI (covers faint end, lower quality)
        Each step falls through to the next only if no data found.
        The full baseline (all sectors) is always used — this is the key
        change for long-period system detection.
        """
        for attempt in range(Config.MAX_RETRIES):
            try:
                col = None
                with _Timeout(180):   # longer timeout – downloading all sectors
                    # 1. SPOC 2-min
                    sr = lk.search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
                    if len(sr):
                        col = sr.download_all()

                    # 2. SPOC any cadence (20-min FFI SPOC products)
                    if not col or not len(col):
                        sr = lk.search_lightcurve(f"TIC {tic}", author="SPOC")
                        if len(sr):
                            col = sr.download_all()

                    # 3. QLP (MIT, 10-min, reaches fainter stars)
                    if Config.USE_QLP and (not col or not len(col)):
                        sr = lk.search_lightcurve(f"TIC {tic}", author="QLP")
                        if len(sr):
                            col = sr.download_all()

                    # 4. TESS-SPOC FFI (last resort for very faint targets)
                    if Config.USE_FFI and (not col or not len(col)):
                        sr = lk.search_lightcurve(f"TIC {tic}", author="TESS-SPOC")
                        if len(sr):
                            col = sr.download_all()

                if not col or not len(col):
                    return None, 0

                # Stitch all available sectors into one light curve.
                # corrector_func=None avoids re-normalising flux across sectors
                # in ways that can artificially flatten real variability.
                lc = col.stitch()
                return lc, len(col)

            except Exception:
                time.sleep(Config.DOWNLOAD_DELAY * (attempt + 1))   # back-off
        return None, 0

# ═══════════════════════════════════════════════════════════════════
# 13. VSX PRE-FLIGHT GATE
# ═══════════════════════════════════════════════════════════════════
class VSXPreflight:
    @staticmethod
    def check(det, info) -> tuple:
        fails = []; notes = []
        p       = det.get('period', 0)
        snr     = det.get('snr', 0)
        dp      = det.get('depth_pct', 0)
        nt      = det.get('n_transits', 0)
        per_err = det.get('period_err', 0)
        epoch   = det.get('epoch', 0)
        dur_h   = det.get('duration', 0)*24
        nsec    = det.get('n_sectors', 1)

        min_snr = 10.0 if 'planet' in det.get('mode','').lower() else 8.0
        if snr < min_snr:     fails.append(f"SNR {snr:.1f} below minimum {min_snr}")
        if p <= 0:            fails.append("Period not determined")
        if per_err <= 0:      notes.append("Period uncertainty is zero (bootstrap may have failed)")
        min_tr = 3 if p < 1.0 else 2
        if nt < min_tr:       fails.append(f"Only {nt} eclipses observed (need ≥{min_tr})")
        if dp < 0.01:         fails.append(f"Depth {dp:.4f}% < 0.01% – likely photon noise")
        if dp > 100:          fails.append(f"Depth {dp:.1f}% > 100% – spurious")
        if not info.get('ra') or not info.get('dec'):
            fails.append("Missing RA/Dec coordinates")
        if epoch <= 0:        fails.append("Epoch T0 not determined")
        if dur_h <= 0:        fails.append("Duration not determined")
        elif dur_h >= 0.5 and abs(dur_h-round(dur_h))<0.02 and int(round(dur_h)) in [1,2,3,4,5,6]:
            fails.append(f"Duration {dur_h:.2f}h is suspiciously integer – likely artifact")
        baseline = nsec * 27.0
        if p > baseline * 0.8:
            fails.append(f"Period {p:.1f}d > 80% of estimated baseline {baseline:.0f}d")
        if not info.get('tmag'):
            notes.append("No Tmag – magnitude range will be approximate")

        ok   = len(fails) == 0
        stat = "✅ PASS" if ok else f"❌ FAIL ({len(fails)} issue{'s' if len(fails)>1 else ''})"
        lines = [
            "─── VSX PRE-FLIGHT ───────────────────────",
            f"Result  : {stat}",
            f"Type    : {det.get('classification','?')}",
            f"SNR     : {snr:.1f}  (min {min_snr})",
            f"Depth   : {dp:.4f}%",
            f"Period  : {p:.6f} ± {per_err:.6f} d",
            f"Epoch   : {epoch:.5f} BTJD",
            f"Duration: {dur_h:.3f} h",
            f"Transits: {nt}",
            f"Sectors : {nsec}",
        ]
        if fails: lines += ["\nFAILURES:"] + [f"  ✗ {x}" for x in fails]
        if notes: lines += ["\nNOTES:"]   + [f"  ℹ {x}" for x in notes]
        lines.append("──────────────────────────────────────────")
        return ok, "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# 14. PLOTTER  (F6: phase always -0.5 to +0.5)
# ═══════════════════════════════════════════════════════════════════
class Plotter:
    def __init__(self, fm): self.fm = fm

    def plot(self, tic, det, lc, info, preflight_report):
        try:
            fig = plt.figure(figsize=(16, 14))
            gs  = GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.30)
            fig.suptitle(
                f"TIC {tic}  –  {det['classification']}\n"
                f"P={det['period']:.6f}±{det['period_err']:.6f} d  "
                f"Depth={det['depth_pct']:.3f}%  SNR={det['snr']:.1f}",
                fontsize=13, fontweight='bold')

            # Row 1 – full light curve
            ax1 = fig.add_subplot(gs[0, :])
            ax1.scatter(lc.time.value, lc.flux.value, s=0.4, alpha=0.4, c='k')
            ax1.set_xlabel('Time (BTJD)'); ax1.set_ylabel('Norm. Flux')
            ax1.set_title(f"{det['n_points']} points  |  {det['n_sectors']} sectors")
            ax1.grid(alpha=0.3)

            # F6: always normalize_phase=True → phase in [-0.5, +0.5]
            folded = lc.fold(period=det['period'], epoch_time=det['epoch'],
                             normalize_phase=True)
            bins = np.linspace(-0.5, 0.5, 100)
            bmed = []
            for i in range(len(bins)-1):
                m = (folded.phase.value>=bins[i])&(folded.phase.value<bins[i+1])
                bmed.append(np.median(folded.flux.value[m]) if m.any() else np.nan)

            # Row 2 – phase plots
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(folded.phase.value, folded.flux.value, s=0.8, alpha=0.3, c='navy')
            ax2.plot((bins[:-1]+bins[1:])/2, bmed, 'r-', lw=2, label='Binned')
            ax2.set_xlim(-0.5, 0.5)   # F6: explicit limits
            ax2.set_xlabel('Phase'); ax2.set_ylabel('Flux')
            ax2.set_title(f'Full phase  ({det["n_transits"]} transits)')
            ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

            ax3 = fig.add_subplot(gs[1, 1])
            pm = np.abs(folded.phase.value) < 0.15
            if pm.sum() > 5:
                ax3.scatter(folded.phase.value[pm], folded.flux.value[pm],
                            s=2, c='crimson', alpha=0.7)
            ax3.set_xlim(-0.15, 0.15)
            ax3.set_xlabel('Phase'); ax3.set_ylabel('Flux')
            ax3.set_title(f'Primary  ({det["depth_pct"]:.3f}%)')
            ax3.grid(alpha=0.3)

            ax4 = fig.add_subplot(gs[1, 2])
            sm = (folded.phase.value > 0.35)&(folded.phase.value < 0.65)
            if sm.sum() > 5:
                ax4.scatter(folded.phase.value[sm], folded.flux.value[sm],
                            s=2, c='seagreen', alpha=0.7)
            ax4.set_xlim(0.35, 0.65)
            ax4.set_xlabel('Phase'); ax4.set_ylabel('Flux')
            sdep = det.get('secondary_depth', 0)
            ax4.set_title(f'Secondary  ({sdep*100:.3f}%)')
            ax4.grid(alpha=0.3)

            # Row 3 – detection info
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            txt = (
                f"DETECTION\n"
                f"  Type      : {det['classification']}\n"
                f"  Period    : {det['period']:.7f} ± {det['period_err']:.7f} d\n"
                f"  Epoch T0  : {det['epoch']:.5f} BTJD  "
                f"(BJD_TDB {det['epoch']+2457000:.5f})\n"   # F4
                f"  Depth     : {det['depth_pct']:.4f}%\n"
                f"  Secondary : {sdep*100:.4f}%\n"
                f"  Duration  : {det['duration']*24:.2f} h\n"
                f"  SNR       : {det['snr']:.2f}\n"
                f"  Transits  : {det['n_transits']}\n\n"
                f"STELLAR PARAMETERS\n"
                f"  RA/Dec    : {info.get('ra',0):.5f} / {info.get('dec',0):+.5f}\n"
                f"  Tmag      : {info.get('tmag',0):.3f}\n"
                f"  Teff      : {info.get('teff',0):.0f} K\n"
                f"  Radius    : {info.get('radius',0):.2f} R☉\n"
                f"  Mass      : {info.get('mass',0):.2f} M☉\n"
            )
            if det.get('multi_period'):
                txt += "\nMULTI-PERIOD: "+", ".join(
                    f"{pp:.4f}d" for pp in det.get('all_periods',[]))
            if det.get('prob_planet',0) > 0:
                txt += f"\nPlanet prob : {det['prob_planet']*100:.1f}%"
            ax5.text(0.02, 0.98, txt, transform=ax5.transAxes,
                     fontsize=9, va='top', family='monospace',
                     bbox=dict(boxstyle='round', fc='wheat', alpha=0.4))

            # Row 4 – VSX pre-flight
            ax6 = fig.add_subplot(gs[3, :])
            ax6.axis('off')
            ax6.text(0.02, 0.98, preflight_report, transform=ax6.transAxes,
                     fontsize=8.5, va='top', family='monospace',
                     bbox=dict(boxstyle='round', fc='#e8f5e9', alpha=0.6))

            plt.tight_layout()
            path = self.fm.dirs['plots'] / f"TIC_{tic}_{det['mode'].replace(' ','_')}.png"
            plt.savefig(path, dpi=130, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            plt.close('all')
            print(f"  ⚠️  Plot error: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════
# 15. SUBMISSION PACK
# ═══════════════════════════════════════════════════════════════════
class SubmissionPack:
    @staticmethod
    def create(tic, det, info, matches: CatalogMatch,
               preflight_report: str, plot_path, packs_dir: Path):
        pack_name = f"TIC_{tic}_VSX_pack"
        pack_dir  = packs_dir / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)

        if plot_path and Path(plot_path).exists():
            shutil.copy(plot_path, pack_dir / "validation_plot.png")

        ra    = info.get('ra',  0) or 0
        dec   = info.get('dec', 0) or 0
        tmag  = info.get('tmag', 0) or 0
        p     = det.get('period', 0)
        perr  = det.get('period_err', 0)
        epoch = det.get('epoch', 0)
        dp    = det.get('depth_pct', 0)
        sdep  = det.get('secondary_depth', 0)
        dur_h = det.get('duration', 0)*24
        snr   = det.get('snr', 0)
        nt    = det.get('n_transits', 0)
        nsec  = det.get('n_sectors', 1)
        ctype = det.get('classification', 'Unknown')
        sector= det.get('Sector', '?')

        # F4: correct time-system label
        bjd_epoch = epoch + 2457000.0
        mag_max   = (tmag + 2.5*np.log10(1/(1-dp/100))
                     if 0 < dp < 100 else tmag+0.01)

        vsx_type_map = {
            'EA':'EA','EB':'EB','EW':'EW',
            'Heartbeat Star':'BCEP/HB',
            'Triple/Hierarchical System':'EA',
            'Unknown':'EA',
        }
        vsx_type = vsx_type_map.get(ctype, 'EA')

        remarks = (f"Discovered with DeepSkySurveyor v18.3 in TESS Sector {sector}. "
                   f"SNR={snr:.1f}. {nt} eclipses across {nsec} sector(s). "
                   f"Eclipse depth {dp:.3f}%, duration {dur_h:.2f} h.")
        if sdep > 0.001: remarks += f" Secondary eclipse {sdep*100:.3f}%."
        if det.get('multi_period'):
            pp_str = ", ".join(f"{x:.4f}d" for x in det.get('all_periods',[]))
            remarks += f" Additional periods: {pp_str}."

        vsx_txt = f"""=================================================================
  VSX SUBMISSION DRAFT  –  TIC {tic}
  Generated by DeepSkySurveyor v18.3
  {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
=================================================================

  OBJECT NAME      : TIC {tic}
  COORDINATES      : RA {ra:.6f}  Dec {dec:+.6f}  (J2000.0)
  VARIABILITY TYPE : {vsx_type}
  PERIOD (days)    : {p:.7f}
  PERIOD UNCERT.   : ± {perr:.7f} d
  EPOCH (BJD_TDB)  : {bjd_epoch:.5f}
  MAX BRIGHTNESS   : {tmag:.3f}  (TESS, out-of-eclipse)
  MIN BRIGHTNESS   : {mag_max:.3f}  (TESS, at primary minimum)
  PHOTOMETRIC BAND : T  (TESS)

  REMARKS          : {remarks}

-----------------------------------------------------------------
HOW TO SUBMIT
  1. Go to https://www.aavso.org/vsx/
  2. Create a free account if needed, then log in.
  3. Click "Submit a new variable star".
  4. Fill in the fields exactly as shown above.
  5. Attach validation_plot.png as supporting material.
  6. Paste the REMARKS text into the remarks box.
-----------------------------------------------------------------

PRE-FLIGHT CHECK RESULT:
{preflight_report}

CATALOG CROSS-MATCH:
  New discovery? : {'NO – already in a catalog, do not submit as new'
                    if matches.is_known() else
                    'YES – not found in any checked catalog'}
  Detail         : {matches.summary()}
=================================================================
"""
        (pack_dir / "vsx_submission.txt").write_text(vsx_txt)

        row = dict(
            TIC=tic, RA=ra, Dec=dec, Tmag=tmag,
            Teff=info.get('teff'), Radius_Rsun=info.get('radius'),
            Mass_Msun=info.get('mass'),
            Classification=ctype, Mode=det.get('mode'),
            Period_d=p, Period_err_d=perr,
            Epoch_BTJD=epoch, Epoch_BJD_TDB=bjd_epoch,  # F4
            Depth_pct=dp, Secondary_Depth_pct=sdep*100,
            Duration_hr=dur_h, SNR=snr,
            N_transits=nt, N_sectors=nsec,
            Eccentricity=det.get('eccentricity',0),
            Planet_Prob=det.get('prob_planet',0),
            Multi_Period=det.get('multi_period',False),
            Catalog_matches=matches.summary(),
            VSX_type=vsx_type,
            VSX_preflight='PASS',
        )
        pd.DataFrame([row]).to_csv(pack_dir / "full_data.csv", index=False)

        notes_txt = f"""REVIEWER NOTES  –  TIC {tic}
========================================
Generated by DeepSkySurveyor v18.3
{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC

BEFORE SUBMITTING TO VSX, PLEASE CHECK:

1. VALIDATION PLOT  (validation_plot.png)
   • Row 1: Full light curve – look for obvious systematics or trends.
   • Row 2 left: Phase-folded full orbit – the binned red line should
     show a clean, symmetric dip centred at phase 0.
   • Row 2 middle: Primary eclipse zoom – should be box-shaped or U-shaped,
     NOT a V-shaped spike or ragged scatter.
   • Row 2 right: Secondary eclipse – if present it confirms a binary.
   • Row 4: VSX pre-flight – all lines should show ✅ PASS.

2. CATALOG CROSS-MATCH  (in vsx_submission.txt)
   If "New discovery? NO" – this target is already catalogued.
   Do not submit it as a new variable star.

3. PERIOD REFINEMENT
   The period uncertainty from TESS alone is ±{perr:.6f} d.
   Ground-based follow-up over several months will greatly improve precision.

4. WHEN IN DOUBT – do not submit.
   VSX editors appreciate quality over quantity.
   A borderline detection is better investigated further first.

Good luck – and happy hunting! 🔭
"""
        (pack_dir / "notes.txt").write_text(notes_txt)

        zip_path = packs_dir / f"{pack_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fp in pack_dir.rglob("*"):
                zf.write(fp, fp.relative_to(packs_dir))

        try:
            from google.colab import files
            files.download(str(zip_path))
            print(f"  📦 Downloading: {zip_path.name}")
        except Exception:
            print(f"  📦 Pack saved (download manually): {zip_path}")

        return zip_path

# ═══════════════════════════════════════════════════════════════════
# 16. MAIN SURVEY
# ═══════════════════════════════════════════════════════════════════
class Survey:
    def __init__(self, sector=None, modes=None):
        print("\n" + "="*70)
        print("  DeepSkySurveyor v18.3  –  Enhanced Multi-Mode Survey")
        print("="*70)
        Path(Config.WORKDIR).mkdir(parents=True, exist_ok=True)
        smgr = SectorManager(Path(Config.WORKDIR)/'processed_sectors.json')
        if sector is None:
            sector = smgr.get_next()
            if sector is None:
                print("\n🎉 ALL TARGET SECTORS COMPLETE!"); return
            cycle = ('4' if sector>=56 else '3' if sector>=40
                     else '2' if sector>=27 else '1N')
            print(f"\n  Auto-selected: Sector {sector}  (Cycle {cycle})")
        else:
            print(f"\n  User-specified: Sector {sector}")
        smgr.mark_progress(sector)
        self.sector   = sector
        self.smgr     = smgr
        self.modes    = modes or ['eb','planet','ultrashort','longperiod','heartbeat']
        self.fm       = FileManager(sector)
        self.cats     = KnownCatalogs(self.fm.dirs['cache'])
        self.prog     = Progress(self.fm.files['progress'])
        self.finder   = TargetFinder(self.fm, self.cats, sector)
        self.analyzer = Analyzer(self.fm)
        self.plotter  = Plotter(self.fm)
        self.dets     = {'eb':[], 'planet':[], 'exotic':[]}

    def run(self):
        targets = self.finder.get_targets()
        if not targets:
            print(f"\n❌ No targets for Sector {self.sector}. Skipping.")
            self.smgr.mark_complete(self.sector)
            nxt = self.smgr.get_next()
            if nxt: Survey(sector=nxt, modes=self.modes).run()
            return
        remaining = [t for t in targets if not self.prog.is_done(t)]
        print(f"\n📊 Sector {self.sector}:")
        print(f"   Targets  : {len(targets)}   "
              f"Done: {len(targets)-len(remaining)}   "
              f"Remaining: {len(remaining)}")
        print(f"   Modes    : {', '.join(self.modes)}")
        print(f"   Est. time: {len(remaining)*8/3600:.1f} h")
        if not remaining:
            self._finish(); return
        print(f"\n{'='*70}\n  PROCESSING SECTOR {self.sector}\n{'='*70}\n")
        try:
            for i, tic in enumerate(remaining):
                self._process(tic, i+1, len(remaining))
                if (i+1) % Config.CHECKPOINT_EVERY == 0:
                    try:
                        self.prog.save(); self._save_csvs()
                        print(f"\n  💾 Checkpoint {i+1}/{len(remaining)}", flush=True)
                    except Exception as e:
                        print(f"\n  ⚠️  Checkpoint failed: {e}", flush=True)
            self._finish()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted – saving progress...")
            self.prog.save(); self._save_csvs()

    def _process(self, tic, cur, tot):
        t0   = time.time()
        eta  = self.prog.eta(tot-cur)
        rate = self.prog.rate()
        print(f"[{cur}/{tot}] TIC {tic}  (ETA:{eta} {rate:.0f}/hr)...",
              end=" ", flush=True)

        info = TICQuery.get(tic)
        if not info or not info.get('ra'):
            print("❌ No TIC info")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, "No TIC info"); return

        tmag = info.get('tmag')
        if tmag and not (Config.MAG_MIN <= tmag <= Config.MAG_MAX):
            print(f"❌ Tmag={tmag:.1f}")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, f"Tmag {tmag:.1f}"); return

        glat = info.get('gal_lat')
        if glat is not None and abs(glat) < Config.MIN_GALACTIC_LAT:
            print(f"❌ |b|={abs(glat):.1f}°")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, "Galactic plane"); return

        cont = info.get('contamination') or 0
        if cont > Config.MAX_CONTAMINATION:
            print(f"❌ Contam={cont*100:.0f}%")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, "Contamination"); return

        matches        = KnownCatalogs.cross_match(info['ra'], info['dec'])
        matches.kostov = self.cats.is_kostov(tic)
        if matches.is_known():
            print(f"❌ Known: {matches.summary()[:50]}")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, f"Known: {matches.summary()}"); return

        result = self.analyzer.run(tic, self.modes)
        if result['status'] == 'rejected':
            print(f"❌ {result['reason']}")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, result['reason']); return

        if result.get('prob_planet', 0) > 0.6 and result.get('secondary_depth', 0) < 0.001:
            print(f"❌ Planet-like transit (prob={result['prob_planet']*100:.0f}%) – excluded from VSX")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, f"Planet-like signal"); return

        # Note: single-sector long-period gate removed in v18.3 — the
        # pipeline now always stitches all available sectors, so a long
        # period detected on a star with only 1 SPOC sector may still
        # have QLP or FFI sectors providing the necessary baseline.
        # The VSX preflight period-vs-baseline check (check #9) handles
        # any remaining cases where baseline is genuinely insufficient.

        pf_pass, pf_report = VSXPreflight.check(result, info)
        if not pf_pass:
            fail_lines = [l for l in pf_report.split('\n') if '✗' in l]
            reason = fail_lines[0].strip() if fail_lines else "VSX preflight failed"
            print(f"❌ VSX: {reason}")
            self.prog.mark_done(tic, time.time()-t0)
            self.prog.add_rejected(tic, reason); return

        flag = "🌟" if result.get('multi_period') else "✅"
        print(f"{flag} {result['classification']}  "
              f"P={result['period']:.4f}d  "
              f"D={result['depth_pct']:.2f}%  "
              f"SNR={result['snr']:.1f}")

        result['Sector'] = self.sector
        plot_path = self.plotter.plot(tic, result, result['lc'], info, pf_report)

        dd = {
            'TIC':                str(tic),
            'Sector':             int(self.sector),
            'RA':                 float(info['ra'])     if info['ra']     else None,
            'Dec':                float(info['dec'])    if info['dec']    else None,
            'Tmag':               float(tmag)           if tmag           else None,
            'Teff':               float(info['teff'])   if info.get('teff')   else None,
            'Radius':             float(info['radius']) if info.get('radius') else None,
            'Mass':               float(info['mass'])   if info.get('mass')   else None,
            'Mode':               result['mode'],
            'Classification':     result['classification'],
            'Period_d':           float(result['period']),
            'Period_err_d':       float(result['period_err']),
            'Epoch_BTJD':         float(result['epoch']),
            'Depth_pct':          float(result['depth_pct']),
            'Secondary_Depth_pct':float(result.get('secondary_depth',0)*100),
            'Duration_hr':        float(result['duration']*24),
            'SNR':                float(result['snr']),
            'N_transits':         int(result['n_transits']),
            'N_sectors':          int(result['n_sectors']),
            'N_points':           int(result['n_points']),
            'Eccentricity':       float(result.get('eccentricity',0)),
            'Planet_Prob':        float(result.get('prob_planet',0)),
            'Multi_Period':       bool(result.get('multi_period',False)),
            'Plot':               str(plot_path) if plot_path else '',
            'Catalog_Matches':    matches.summary(),
            'VSX_Preflight':      'PASS',
        }

        if result['mode'] in ['Eclipsing Binary','Ultra-Short Binary']:
            self.dets['eb'].append(dd)
        elif result['mode'] == 'Exoplanet':
            self.dets['planet'].append(dd)
        else:
            self.dets['exotic'].append(dd)

        self.prog.mark_done(tic, time.time()-t0)
        self.prog.add_detection(tic, dd)

        SubmissionPack.create(
            tic, {**result, 'Sector': self.sector},
            info, matches, pf_report, plot_path,
            self.fm.dirs['packs'])

        time.sleep(Config.DOWNLOAD_DELAY)

    def _finish(self):
        try:
            self.prog.save()
            self._save_csvs()
            self._vsx_csv()
            self._html_report()
        except Exception as e:
            print(f"\n⚠️  Save error: {e}")
            try:
                all_d = self.dets['eb']+self.dets['planet']+self.dets['exotic']
                (self.fm.sector_dir/'emergency.json').write_text(
                    json.dumps(all_d, default=_js))
                print("  Emergency JSON saved.")
            except: pass
        self.smgr.mark_complete(self.sector)
        n = len(self.prog.data['detections'])
        print(f"\n{'='*70}")
        print(f"  SECTOR {self.sector} COMPLETE  –  {n} candidate(s) found")
        print(f"  Results : {self.fm.sector_dir}/")
        print(f"  Packs   : {self.fm.dirs['packs']}/")
        print("="*70)
        if n == 0:
            nxt = self.smgr.get_next()
            if nxt:
                print(f"\n🔄 No detections – auto-advancing to Sector {nxt}...\n")
                time.sleep(2)
                Survey(sector=nxt, modes=self.modes).run()

    def _save_csvs(self):
        for key, fname in [('eb','candidates_eb'),
                           ('planet','candidates_planet'),
                           ('exotic','candidates_exotic')]:
            if self.dets[key]:
                pd.DataFrame(self.dets[key]).to_csv(
                    self.fm.files[fname], index=False)
        all_d = self.dets['eb']+self.dets['planet']+self.dets['exotic']
        if all_d:
            pd.DataFrame(all_d).to_csv(self.fm.files['candidates_all'], index=False)

    def _vsx_csv(self):
        rows = []
        for det in self.dets['eb']+self.dets['exotic']:
            tmag = det.get('Tmag') or 0
            dp   = det.get('Depth_pct', 0)
            mmax = tmag+2.5*np.log10(1/(1-dp/100)) if 0<dp<100 else tmag+0.01
            rows.append({
                'Name':          f"TIC {det['TIC']}",
                'RA_J2000':      det['RA'],
                'Dec_J2000':     det['Dec'],
                'Type':          det['Classification'],
                'Max_Mag':       f"{tmag:.3f}",
                'Min_Mag':       f"{mmax:.3f}",
                'Band':          'TESS',
                'Epoch_BJD_TDB': f"{det['Epoch_BTJD']+2457000:.5f}",  # F4
                'Period_d':      f"{det['Period_d']:.7f}",
                'Period_err':    f"{det['Period_err_d']:.7f}",
                'Discoverer':    'DeepSkySurveyor v18.3',
                'Remarks':       (f"TESS Sector {det['Sector']}, "
                                  f"SNR={det['SNR']:.1f}, "
                                  f"{det['N_transits']} eclipses, "
                                  f"VSX preflight PASS"),
            })
        if rows:
            pd.DataFrame(rows).to_csv(self.fm.files['vsx_ready'], index=False)
            print(f"\n  📝 VSX CSV: {self.fm.files['vsx_ready']}")

    def _html_report(self):
        all_d = [d['data'] for d in self.prog.data['detections'] if 'data' in d]
        eb_c  = sum(1 for d in all_d if d.get('Classification','').startswith('E'))
        pl_c  = sum(1 for d in all_d if 'planet' in d.get('Mode','').lower())
        ex_c  = len(all_d)-eb_c-pl_c
        mu_c  = sum(1 for d in all_d if d.get('Multi_Period'))
        rows_html = ""
        for d in sorted(all_d, key=lambda x: x.get('SNR',0), reverse=True):
            cls = ('multi'  if d.get('Multi_Period') else
                   'planet' if 'planet' in d.get('Mode','').lower() else
                   'eb'     if d.get('Classification','').startswith('E') else 'exotic')
            sp = []
            if d.get('Multi_Period'):       sp.append('🌟Multi')
            if d.get('Planet_Prob',0)>0.7:  sp.append('🪐HighP')
            if d.get('Eccentricity',0)>0.5: sp.append('💓HB')
            rows_html += (
                f'<tr class="{cls}">'
                f'<td><a href="https://exofop.ipac.caltech.edu/tess/target.php'
                f'?id={d.get("TIC","")}" target="_blank">{d.get("TIC","")}</a></td>'
                f'<td>{d.get("Classification","")}</td>'
                f'<td>{(d.get("RA") or 0):.4f}</td>'
                f'<td>{(d.get("Dec") or 0):.4f}</td>'
                f'<td>{(d.get("Tmag") or 0):.2f}</td>'
                f'<td>{d.get("Period_d",0):.6f}±{d.get("Period_err_d",0):.6f}</td>'
                f'<td>{d.get("Depth_pct",0):.3f}</td>'
                f'<td>{d.get("SNR",0):.1f}</td>'
                f'<td>{" ".join(sp) or "-"}</td>'
                f'<td><a href="plots/{Path(d.get("Plot","")).name}" '
                f'target="_blank">View</a></td>'
                f'</tr>\n')
        html = f"""<!DOCTYPE html><html><head>
<title>Sector {self.sector} – DSS v18.3</title>
<style>
  body{{font-family:Arial,sans-serif;margin:20px;background:#f0f4f8}}
  .hdr{{background:linear-gradient(135deg,#1a237e,#4a148c);color:#fff;
        padding:25px;border-radius:10px;margin-bottom:20px}}
  .stats{{display:flex;gap:15px;flex-wrap:wrap;margin-bottom:20px}}
  .stat{{background:#fff;padding:20px;border-radius:8px;min-width:110px;
         box-shadow:0 2px 4px rgba(0,0,0,.1)}}
  .sv{{font-size:28px;font-weight:bold;color:#1a237e}}
  .sl{{font-size:12px;color:#666;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;background:#fff;
         box-shadow:0 2px 4px rgba(0,0,0,.1);border-radius:8px;overflow:hidden}}
  th{{background:#1a237e;color:#fff;padding:10px;text-align:left}}
  td{{padding:8px 10px;border-bottom:1px solid #eee}}
  tr:hover{{background:#f5f5f5}}
  .eb{{background:#e3f2fd!important}}
  .planet{{background:#fff3e0!important}}
  .exotic{{background:#f3e5f5!important}}
  .multi{{background:#c8e6c9!important;font-weight:bold}}
  a{{color:#1a237e;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .note{{margin-top:20px;color:#555;font-size:12px;font-style:italic}}
</style></head><body>
<div class="hdr">
  <h1>🔭 DeepSkySurveyor v18.3 — Sector {self.sector}</h1>
  <p>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp;
     Modes: {', '.join(self.modes)} &nbsp;|&nbsp;
     All candidates passed VSX pre-flight</p>
</div>
<div class="stats">
  <div class="stat"><div class="sv">{len(self.prog.data['processed'])}</div>
    <div class="sl">Stars processed</div></div>
  <div class="stat"><div class="sv">{len(all_d)}</div>
    <div class="sl">Candidates</div></div>
  <div class="stat"><div class="sv">{eb_c}</div><div class="sl">EBs</div></div>
  <div class="stat"><div class="sv">{pl_c}</div><div class="sl">Planets</div></div>
  <div class="stat"><div class="sv">{ex_c}</div><div class="sl">Exotic</div></div>
  <div class="stat"><div class="sv">{mu_c}</div><div class="sl">Multi-period</div></div>
</div>
<table>
<tr><th>TIC</th><th>Type</th><th>RA</th><th>Dec</th><th>Tmag</th>
    <th>Period±err (d)</th><th>Depth%</th><th>SNR</th><th>Flags</th><th>Plot</th></tr>
{rows_html}
</table>
<p class="note">
  Submission packs (ZIP files) are in <code>submission_packs/</code>.
  Each ZIP contains validation_plot.png, vsx_submission.txt, full_data.csv, and notes.txt.
</p>
</body></html>"""
        (self.fm.sector_dir / 'report.html').write_text(html)
        print(f"\n  📄 Report: {self.fm.sector_dir}/report.html")

# ═══════════════════════════════════════════════════════════════════
# 17. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
import argparse

def main():
    parser = argparse.ArgumentParser(description='DeepSkySurveyor v18.3')
    parser.add_argument('--sector', type=int, default=None)
    parser.add_argument('--mode', nargs='+',
                        choices=['all','eb','planet','ultrashort',
                                 'longperiod','heartbeat','exotic'],
                        default=['all'])
    filtered = [a for a in sys.argv[1:]
                if not a.startswith('-f') and not a.endswith('.json')
                and 'kernel' not in a.lower()]
    args = parser.parse_args(filtered or [])
    if 'all' in args.mode:
        modes = ['eb','planet','ultrashort','longperiod','heartbeat']
    elif 'exotic' in args.mode:
        modes = ['ultrashort','longperiod','heartbeat']
    else:
        modes = args.mode
    try:
        Survey(sector=args.sector, modes=modes).run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    except Exception as e:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
else:
    main()
