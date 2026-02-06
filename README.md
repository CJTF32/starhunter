# 🔭 DeepSkySurveyor v17

**Autonomous Multi-Mode Discovery Engine for TESS Data**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TESS Mission](https://img.shields.io/badge/TESS-Mission-red.svg)](https://tess.mit.edu/)

An intelligent, automated pipeline for discovering eclipsing binaries, exoplanets, and exotic stellar systems in TESS data. Generates VSX-ready submissions with comprehensive validation.

![DeepSkySurveyor Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=DeepSkySurveyor+v17)

---

## 🌟 Features

### Multi-Mode Detection
- **Eclipsing Binaries** (EA/EB/EW classification)
- **Exoplanet Transits** (with planet probability scoring)
- **Ultra-Short Period Binaries** (< 0.3 days)
- **Long-Period Systems** (15-60 days)
- **Heartbeat Stars** (eccentric binaries with tidal distortion)
- **Multi-Period Systems** (circumbinary planets, triple stars)

### Intelligent Processing
- ✅ **Auto-starts at optimal sector** (Sector 14 - high target density)
- ✅ **Auto-skips empty sectors** - moves to next if no detections
- ✅ **Resume capability** - picks up where it left off
- ✅ **Progress tracking** - saves every 25 targets
- ✅ **Comprehensive validation** - filters artifacts and known variables

### Professional Output
- 📊 **VSX-ready CSV** - submit directly to Variable Star Index
- 📈 **Beautiful diagnostic plots** - multi-panel light curve analysis
- 🎨 **Interactive HTML reports** - sortable, filterable results
- 📁 **Organized file structure** - separate files by detection type

### Quality Assurance
- ✅ **Multi-catalog cross-matching** (VSX, ASAS-SN, ATLAS, Gaia DR3)
- ✅ **Automated EB classification** (EA/EB/EW types)
- ✅ **Bootstrap period uncertainties** - proper error estimation
- ✅ **Artifact rejection** (sidereal day, TESS orbit, instrumental aliases)
- ✅ **Minimum SNR 7.0** - high-confidence detections only

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepSkySurveyor.git
cd DeepSkySurveyor

# Run the script (auto-installs dependencies)
python DeepSkySurveyor_v17_TESTED.py
```

That's it! The script will:
1. Auto-install all dependencies (numpy, pandas, lightkurve, astropy, etc.)
2. Start at Sector 14 (optimal target density)
3. Process all targets
4. Generate reports and VSX submissions
5. Auto-skip to next sector if no detections found

### Requirements

- Python 3.8+
- Internet connection (for MAST/TESS data downloads)
- ~5GB disk space per sector
- Recommended: 8GB+ RAM

**Dependencies** (auto-installed):
- `numpy`, `pandas`, `matplotlib`
- `scipy`, `lightkurve`
- `astropy`, `astroquery`
- `requests`

---

## 📖 Usage

### Basic Usage (Recommended)

```bash
# Run with default settings (starts Sector 14, all modes)
python DeepSkySurveyor_v17_TESTED.py
```

### Detection Modes

```bash
# Eclipsing binaries only (fastest, most common)
python DeepSkySurveyor_v17_TESTED.py --mode eb

# Exoplanets only
python DeepSkySurveyor_v17_TESTED.py --mode planet

# Exotic systems (ultra-short, heartbeat, long-period)
python DeepSkySurveyor_v17_TESTED.py --mode exotic

# All detection modes
python DeepSkySurveyor_v17_TESTED.py --mode all

# Custom combination
python DeepSkySurveyor_v17_TESTED.py --mode eb planet ultrashort
```

### Sector Selection

```bash
# Specific sector
python DeepSkySurveyor_v17_TESTED.py --sector 26

# With mode selection
python DeepSkySurveyor_v17_TESTED.py --sector 14 --mode eb
```

### In Google Colab

```python
# Upload the script to Colab, then:
!python DeepSkySurveyor_v17_TESTED.py --mode eb
```

---

## 📊 Output Files

After running, check the output directory:

```
DeepSkySurveyor_v17/
├── processed_sectors.json          # Progress tracker
└── Sector_14/
    ├── report.html                 # ⭐ Interactive report (open in browser)
    ├── vsx_submissions.csv         # ⭐ VSX-ready submissions
    ├── candidates_eb.csv           # Eclipsing binaries
    ├── candidates_planet.csv       # Planet candidates
    ├── candidates_exotic.csv       # Exotic systems
    ├── candidates_all.csv          # All detections combined
    ├── progress.json               # Processing status
    ├── rejected.csv                # Filtered targets
    └── plots/                      # Diagnostic plots
        ├── TIC_12345678_Eclipsing_Binary.png
        └── TIC_87654321_Exoplanet.png
```

### Key Files

**`vsx_submissions.csv`** - Ready to submit to [VSX](https://www.aavso.org/vsx/)
- Contains all required fields
- Period with uncertainties
- Magnitude ranges
- Classification (EA/EB/EW)
- Discovery information

**`report.html`** - Beautiful interactive report
- Color-coded by detection type
- Sortable by SNR, period, depth
- Links to ExoFOP
- Links to diagnostic plots

**`candidates_all.csv`** - Complete detection list
- All parameters
- Stellar properties
- Multi-period flags
- Planet probabilities

---

## 🎯 Expected Results

### Sector 14 (Default Starting Point)
- **Targets:** ~15,000 stars
- **Runtime:** 20-40 hours (full processing)
- **Expected Discoveries:**
  - 50-150 eclipsing binaries (EA/EB/EW)
  - 5-20 planet candidates
  - 2-10 exotic systems (ultra-short, heartbeat)
  - 0-3 multi-period systems (rare!)

### Quality Distribution
- **SNR > 15:** ~20% (excellent, publication-ready)
- **SNR 10-15:** ~30% (very good, submit confidently)
- **SNR 7-10:** ~50% (good, review plot carefully)

---

## 🔬 Detection Methods

### Eclipsing Binaries

Uses **Box Least Squares (BLS)** periodogram:
- Period search: 0.3 - 15 days
- Minimum SNR: 7.0
- Automatic classification:
  - **EA (Algol):** Detached, deep primary eclipse
  - **EB (β Lyrae):** Semi-detached, both eclipses visible
  - **EW (W UMa):** Contact binary, equal eclipses

### Exoplanets

Optimized for small planets:
- Period search: 0.5 - 100 days
- Minimum depth: 0.01% (Earth-sized detectable)
- **Odd/even transit test** - validates consistency
- **Planet probability score** - distinguishes from binaries

### Multi-Period Detection

After finding primary signal:
1. Mask primary transits
2. Search residuals for secondary periods
3. Check for harmonic relationships
4. Flag as triple system or circumbinary planet

---

## 🧪 Validation Pipeline

### 1. Initial Filters
- Magnitude: 9.0 - 14.0 (TESS mag)
- Galactic latitude: |b| > 10° (avoid crowded fields)
- Contamination ratio: < 10%
- Minimum data points: 1000

### 2. Signal Detection
- BLS periodogram search
- Minimum SNR: 7.0
- Minimum transits: 2-3 (depends on mode)

### 3. Artifact Rejection
Rejects:
- Sidereal day aliases (0.997/n days)
- Day aliases (0.5, 1.0, 2.0 days)
- TESS orbit aliases (13.7, 6.85 days)

### 4. Cross-Matching
Checks against:
- **VSX** - Variable Star Index
- **ASAS-SN** - All-Sky Automated Survey
- **ATLAS** - ATLAS Variable Stars
- **Gaia DR3** - Gaia Variables
- **Kostov Catalog** - Known TESS EBs

### 5. Classification
- EB type determination (EA/EB/EW)
- Planet probability scoring
- Heartbeat detection
- Multi-period analysis

---

## 📈 Performance

### Speed
- **~5-10 seconds per target** (with all modes)
- **~2-3 seconds per target** (EB mode only)
- Checkpoint saves every 25 targets

### Resource Usage
- **Memory:** ~2-4 GB RAM
- **Disk:** ~5 GB per sector
- **Network:** Downloads TESS light curves on-demand

### Optimization Tips

**For faster processing:**
```bash
# EB mode only (2x faster)
python DeepSkySurveyor_v17_TESTED.py --mode eb

# Narrow magnitude range (fewer targets)
# Edit Config.MAG_MIN = 10.0, Config.MAG_MAX = 12.0
```

**For comprehensive survey:**
```bash
# All modes (thorough but slower)
python DeepSkySurveyor_v17_TESTED.py --mode all
```

---

## 🎓 Scientific Use Cases

### Variable Star Discovery
- Submit discoveries to [VSX](https://www.aavso.org/vsx/)
- CSV output formatted for direct submission
- All required fields auto-generated

### Exoplanet Follow-Up
- High planet-probability candidates
- Submit to [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/)
- Ground-based follow-up prioritization

### Binary Star Studies
- Contact binary surveys (EW types)
- Ultra-short period systems (< 6 hours)
- Eccentric binaries (heartbeat stars)

### Multi-Stellar Systems
- Circumbinary planet candidates
- Triple star systems
- Hierarchical systems

---

## 🏆 Priority Targets

### What to Look For

**🌟 Multi-Period Systems (Highest Priority)**
- Multi_Period = True in output
- Triple stars or circumbinary planets
- Extremely rare, highly publishable

**⚡ Ultra-Short Period Binaries**
- Period < 0.2 days (< 5 hours)
- SNR > 10
- Rare and scientifically valuable

**💎 Contact Binaries (EW type)**
- SNR > 15
- Clean phase-folded curve
- Always needed for catalogs

**🪐 High-Confidence Planets**
- Planet_Prob > 0.8
- Depth < 2%
- Multiple transits

**💓 Heartbeat Stars**
- Eccentricity > 0.6
- Beautiful tidal distortion signature
- Important for tidal physics

---

## 🐛 Troubleshooting

### Common Issues

**"No targets found in sector"**
- Some sectors have fewer targets
- Script will auto-skip to next sector
- Try: `--sector 14` (guaranteed high density)

**Dependencies fail to install**
```bash
# Pre-install manually:
pip install --upgrade pip
pip install numpy pandas matplotlib scipy
pip install lightkurve astropy astroquery requests
```

**Out of memory in Colab**
```bash
# Reduce magnitude range (fewer targets)
# Edit script: MAG_MIN = 10.0, MAG_MAX = 12.0
```

**Script interrupted**
- Progress is auto-saved every 25 targets
- Just re-run - it will resume automatically

**No detections found**
- Normal for some sectors
- Script will auto-advance to next sector
- Try lowering SNR threshold (edit MIN_SNR in MODES)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional detection modes (RR Lyrae, Cepheids)
- [ ] Machine learning classification
- [ ] Parallel processing for speed
- [ ] Web interface
- [ ] Real-time TESS sector processing

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Documentation

### Detection Parameters

```python
# Eclipsing Binary Mode
min_period = 0.3 days
max_period = 15.0 days
min_depth = 0.3%
min_snr = 7.0

# Exoplanet Mode
min_period = 0.5 days
max_period = 100.0 days
min_depth = 0.01%
min_snr = 7.0

# Ultra-Short Mode
min_period = 0.1 days
max_period = 0.3 days
min_snr = 7.0
```

### Customization

Edit the `Config` class in the script:

```python
class Config:
    MAG_MIN = 9.0       # Minimum TESS magnitude
    MAG_MAX = 14.0      # Maximum TESS magnitude
    MIN_GALACTIC_LAT = 10.0  # Avoid Galactic plane
    MAX_CONTAMINATION = 0.10  # Max contamination ratio
    MIN_DATA_POINTS = 1000    # Min light curve points
```

Edit detection modes in `MODES` dictionary for custom searches.

---

## 📖 Citation

If you use DeepSkySurveyor in your research, please cite:

```bibtex
@software{deepskysurveyor2024,
  author = {Your Name},
  title = {DeepSkySurveyor: Autonomous Discovery Engine for TESS Data},
  year = {2024},
  url = {https://github.com/yourusername/DeepSkySurveyor}
}
```

---

## 🙏 Acknowledgments

- **TESS Mission** - NASA's Transiting Exoplanet Survey Satellite
- **Lightkurve** - Python package for TESS/Kepler data analysis
- **MAST** - Mikulski Archive for Space Telescopes
- **AAVSO/VSX** - American Association of Variable Star Observers
- **Astroquery** - Python package for astronomical data queries

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- [TESS Mission](https://tess.mit.edu/)
- [VSX - Variable Star Index](https://www.aavso.org/vsx/)
- [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/)
- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [MAST Archive](https://mast.stsci.edu/)

---

