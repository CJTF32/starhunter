# 🔭 TESS Variable Star Hunter

A Python pipeline for the automated detection and validation of **Eclipsing Binary** systems using data from NASA's Transiting Exoplanet Survey Satellite (TESS).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

## 📖 Overview

The TESS mission provides time-series photometry for millions of stars. Analyzing this data manually is impossible. This tool acts as a "digital sieve," downloading batches of light curves, processing them to remove stellar variability, and applying **Box Least Squares (BLS)** algorithms to detect periodic transit events.

Key Features:
* **Automated Batch Processing:** Scans hundreds of stars via the MAST API.
* **Systematic Noise Rejection:** Automatically filters out the common ~7-day TESS momentum dump glitches.
* **Alias Checking:** Validates candidates by folding data at $2 \times P$ to distinguish between Primary/Secondary eclipses and identify Contact Binaries.
* **SNR Calculation:** robust Signal-to-Noise Ratio estimation for candidate ranking.

## 🚀 Discoveries

This pipeline has been used to identify previously uncatalogued variable systems.

| TIC ID | Constellation | Period ($P$) | Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **TIC 280794538** | Tucana | 10.65 d | Detached EB | Deep primary eclipse, distinct secondary. |
| **TIC 394105479** | Indus | 6.60 d | Detached EB | High SNR signal (~40% flux drop). |
| **TIC 233066920** | Draco | 1.15 d | W UMa (Contact) | Continuous variation; confirmed via alias check. |

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/TESS-Variable-Star-Hunter.git](https://github.com/YOUR_USERNAME/TESS-Variable-Star-Hunter.git)
