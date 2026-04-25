# Brain State Detection Via EEG: Machine Learning Pipeline

This repository contains a comprehensive machine learning pipeline designed to detect "Mind-Wandering" vs. "Meditation" states using EEG data. While the project involves complex signal processing, this implementation focuses on the **Computer Science** aspects: custom data architectural structures, robust subject-level cross-validation, and high-performance classification using SVM and XGBoost.

---

## 🚀 Quick Start (Running the Implementation)

### 1. Environment Setup

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch mne mne-bids fooof python-dotenv
```

### 2. Data Acquisition

The raw EEG files are several gigabytes. To run the modeling code immediately:

1. **Download** the pre-extracted features from this [Google Drive Folder](https://drive.google.com/drive/folders/1d0I8YHEKWVuzgoq3634raO1OQk_p3IaB?usp=sharing).
2. **Place** the `features` folder into your project directory.
3. **Configure Paths**: Create a `.env` file in the root directory or modify the `ROOT` variable in `03_model.ipynb` to point to your data directory.

### 3. Execution Sequence

For the core CS results (Metrics and Visuals), run the notebooks in this order:

1.  **`data_metrics.ipynb`**: Analyzes class distribution and subject balance.
2.  **`03_model.ipynb`**: The primary engine. It performs data cleaning, subject-level normalization, and trains the classifiers.
    - _Note:_ Ensure the `FEATURES_PATH` variable points to the downloaded folder.
3.  **`04_plot_model_results.ipynb`**: Generates the final comparative visuals, tables, and signal-vs-noise deltas.

**Estimated Time for Completion:** ~15-30 minutes (on `03_model.ipynb`) depending on your CPU, as it utilizes parallel processing (`n_jobs=-1`).

---

## 🛠 Project Architecture

The implementation is divided into two phases: the **EEG Pre-processing Pipeline** (ETL) and the **Machine Learning Pipeline** (Analysis).

### Phase 1: ETL & Feature Engineering (Files 00-02)

_These files handle the domain-specific EEG preparations and are included for completeness._

- **`00_preprocessing.ipynb`**: Automates artifact rejection, bad channel detection (via `PyPrep`), and re-referencing.
- **`01_analysis.ipynb`**: Converts time-series data into the frequency domain using Welch’s Method.
- **`02_feature_extraction.ipynb`**: Uses the **FOOOF** (Fitting Oscillatory & One-Over-F) algorithm to separate aperiodic "brain noise" from periodic neural oscillations.
- **`events_generator.ipynb`**: A utility script to repair synchronization errors found in the original BIDS dataset triggers.
- **`changes_to_raw_data.txt`**: Logs the changes made to the raw data to fix incorrect formatting/labeling.

### Phase 2: Core Machine Learning (Files 03-04 & Metrics)

_This is the primary CS implementation._

- **`data_metrics.ipynb`**: Visualizes the class imbalance per subject (Critical for choosing the `macro-F1` scoring metric).
- **`03_model.ipynb`**:
  - **Custom Dataset Class (`EEGDataset`)**: A robust wrapper that handles subject-level z-score normalization and wide-to-long data pivoting.
  - **Leave-Partial-Group-Out (LPGO)**: A custom cross-validation logic designed to simulate "calibrated" models (training on a portion of a new subject's data).
  - **Classifiers**: Implements `XGBoost` and `RBF-Kernel SVM` with balanced class weights.
- **`04_plot_model_results.ipynb`**:
  - Compares models trained on real labels vs. shuffled labels to prove the model is learning actual brain states rather than just memorizing subject identities.

---

## 📊 Requirements & Libraries

The following Python packages are required:

- `mne` & `mne-bids`: EEG data handling.
- `scikit-learn`: Standard scaling, PCA, SVM, and CV metrics.
- `xgboost`: Gradient boosted decision trees.
- `fooof`: Aperiodic signal parameterization.
- `pandas` & `numpy`: Data manipulation.
- `matplotlib`: Visualization.

---
