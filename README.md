# 🔭 Deep Sky Surveyor

**Automated Variable Star Hunter for TESS FFI Data**

The Deep Sky Surveyor is a robust, resume-capable Python tool designed to scan thousands of stars in the TESS (Transiting Exoplanet Survey Satellite) dataset. It applies Box Least Squares (BLS) algorithms to detect periodic eclipsing binary stars and filters out common instrumental artifacts.

## ✨ Features

* **Whole-Sky Grid Search:** Systematically scans tiles of the sky (RA/DEC grid).
* **Resume Capability:** Uses a JSON checkpoint system. If the script crashes or is stopped, it resumes exactly where it left off.
* **Artifact Filtering:** Aggressively filters out:
    * 1.0-day and 0.5-day aliases (Earthshine/momentum dumps).
    * Grid edge effects.
    * Shallow amplitude noise (< 0.5% depth).
* **Network Resilience:** Includes auto-retry logic for MAST server timeouts.
* **Environment Agnostic:** Works locally or on Google Colab (with auto-download).

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/DeepSkySurveyor.git](https://github.com/yourusername/DeepSkySurveyor.git)
    cd DeepSkySurveyor
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠 Usage

Run the main script:

```bash
python main.py
