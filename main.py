# 🚜 TESS Lawn Mower: Automated Exoplanet Hunter

**An industrial-strength scanning tool for the TESS Northern Continuous Viewing Zone (CVZ).**

This tool performs a systematic "Raster Scan" (grid search) of the North Ecliptic Pole using data from the Transiting Exoplanet Survey Satellite (TESS). Unlike random sampling, this script maps a 10x10 degree grid to ensure 100% coverage of the most data-rich region of the sky.

## 🚀 Key Features

* **Raster Scan Logic:** Systematically moves the telescope search focus in 1-degree steps to prevent gaps.
* **Deep Field Targeting:** Focuses on **Sector 75**, located in the Continuous Viewing Zone (CVZ), ensuring maximum data density and minimal camera gaps.
* **QLP Data Pipeline:** Utilizes the MIT Quick-Look Pipeline (QLP) to access millions of faint stars (Mag 11-14) that standard pipelines miss.
* **Auto-Filtering:**
    * **BLS Algorithm:** Detects periodic dips (transits).
    * **VSX Cross-Match:** Automatically checks the "Variable Star Index" to reject known false positives (eclipsing binaries).
    * **SNR Thresholds:** Only saves candidates with strong signals (Signal-to-Noise Ratio > 30).
* **Resilience:** Includes auto-installers for dependencies and "Connection Retry" logic to handle server timeouts without crashing.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/TESS-LawnMower.git](https://github.com/your-username/TESS-LawnMower.git)
    cd TESS-LawnMower
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

Run the main script. It is fully autonomous.

```bash
python main.py
