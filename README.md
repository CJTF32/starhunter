# 🔭 TESS Variable Star Hunter

A Python pipeline for the automated detection and validation of **Eclipsing Binary** systems using data from NASA's Transiting Exoplanet Survey Satellite (TESS).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📖 Overview

The TESS mission provides time-series photometry for millions of stars. Analyzing this data manually is impossible. This tool acts as a "digital sieve," downloading batches of light curves, processing them to remove stellar variability, and applying **Box Least Squares (BLS)** algorithms to detect periodic transit events.

**Key Features:**
* **Automated Batch Processing:** Scans hundreds of stars via the MAST API without manual intervention.
* **Systematic Noise Rejection:** Automatically identifies and filters out common instrumental artifacts, such as the ~7-day "momentum dump" glitches found in certain TESS sectors.
* **Alias Checking:** Validates candidates by folding data at $2 \times P$ to distinguish between Primary/Secondary eclipses and identify Contact Binaries (W UMa types).
* **SNR Calculation:** Implements robust Signal-to-Noise Ratio estimation to rank candidates by reliability.

## 💾 Data Management

To ensure scientific reproducibility, this pipeline includes an automated backup system.

* **Automatic Storage:** When a strong candidate is detected or validated, the processed light curve (CSV format) and validation plots (PNG format) are automatically saved to a local `data/` directory.
* **Provenance:** These files satisfy requirements for submission to databases like VSX, which may request the specific photometry used to determine the period and epoch.
* **Git Policy:** The `data/` folder is included in `.gitignore` to prevent bloating the repository size, but users are encouraged to archive these files locally.

## 🚀 Discoveries

This pipeline has been successfully used to identify previously uncatalogued variable systems.

| TIC ID | Constellation | Period ($P$) | Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **TIC 280794538** | Tucana | 10.65 d | Detached EB | Deep primary eclipse with distinct secondary dip. |
| **TIC 394105479** | Indus | 6.60 d | Detached EB | High SNR signal (~40% flux drop). |
| **TIC 233066920** | Draco | 1.1582 d | W UMa (Contact) | Continuous ellipsoidal variation; period validated via alias check. |

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/TESS-Variable-Star-Hunter.git](https://github.com/YOUR_USERNAME/TESS-Variable-Star-Hunter.git)
    cd TESS-Variable-Star-Hunter
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

Run the main script to enter the interactive menu:

```bash
python tess_hunter.py
