# Remaining Useful Life (RUL) Prediction for Rolling Bearings

This repository contains the OELP project P157, **Remaining Useful Life Estimation of Bearing Using AI/ML**, carried out by M. Ruthvi (142301021) and G. Trilochan Babu (132301036) under the guidance of Dr. Sagi Rathna Prasad, Department of Mechanical Engineering (May 2026).

The project builds a vibration based condition monitoring pipeline for a rolling element bearing. Raw accelerometer signals are converted into 27 handcrafted features, fused into a Health Indicator (HI) with PCA, used to detect the onset of a fault, diagnosed with FFT against the characteristic bearing defect frequencies, and finally used to train regression models that estimate the bearing's Remaining Useful Life.

## Dataset

Three public run to failure datasets were studied in the exploratory phase: NASA IMS (University of Cincinnati), PRONOSTIA/FEMTO and XJTU‑SY. The NASA IMS dataset was selected for the full pipeline because its degradation trend is the clearest of the three.

The pipeline uses **IMS Test 2, Bearing 1** (channel 0), which is known to fail with an outer race defect. Test 2 contains 984 snapshot files recorded from 2004.02.12 10:32:39 to 2004.02.19 06:22:39. Each file is a one second snapshot of 20,480 samples at 20 kHz, and a new file was recorded every 10 minutes.

The raw data is not included in this repository. The notebooks try to download it automatically through `kagglehub` (dataset `vinayak123tyagi/bearing-dataset`); if that fails, download the IMS bearing data from the NASA Prognostics Data Repository and point `data_dir` in the notebook at the local `2nd_test` folder.

The processed feature table that the models train on is included as `Datasets/rul_features_final_dataset.csv` (402 windows by 35 columns: 27 features plus `time`, `HI`, `HI_top`, `Q_stat`, `fault_label`, `fault_type`, `confidence` and `RUL_true`).

## Method

**Windowing.** The 984 snapshots for Bearing 1 are concatenated into a single signal of 20,152,320 samples. This signal is cut into windows of 100,000 samples (5 s of recorded signal) with a step of 50,000 samples (50% overlap), which gives 402 windows.

**Feature extraction.** Twelve time domain features (RMS, mean, standard deviation, variance, skewness, kurtosis, peak, peak to peak, crest factor, mean absolute value, shape factor, clearance factor), ten frequency domain features (spectral entropy, spectral kurtosis, spectral skewness, peak frequency, spectral RMS, spectral centroid, spectral energy, frequency variance, band power at 1X and 2X) and five time frequency features from a 4 level `db4` wavelet decomposition (wavelet energy, entropy, RMS, kurtosis and variance) are computed for every window, giving 27 features in total.

**Health Indicator.** The first 40% of windows (160) are treated as healthy. Features are Min‑Max scaled on the healthy data, PCA is fitted on the healthy windows, and the first 8 components (about 94.5% of variance) are combined into a Global Principal Component weighted by explained variance. The absolute GPC, normalised to the range 0 to 1, is the HI. Feature contribution scores from the PCA loadings were then used to select the five most informative features (BandPower1X, Spectral Entropy, Spectral Centroid, BandPower2X, Shape Factor), and the HI was rebuilt from those features alone (`HI_top`).

**Fault onset detection.** A threshold of mean plus three standard deviations of the healthy HI is used. With all features the threshold is about 0.002; with the top five features it rises to 0.0695, which separates healthy and faulty behaviour far more clearly. The first crossing after the healthy region is at window 214, which corresponds to t = 537.5 s on the concatenated time axis. A PCA Q statistic (squared prediction error, 99th percentile control limit on healthy data) independently confirms the same onset region. K‑Means clustering is used to separate healthy, intermediate and severe degradation regions.

**Fault diagnosis.** FFT spectra of the degradation windows are compared against the theoretical defect frequencies of the Rexnord ZA‑2115 bearing (BPFO ≈ 236.4 Hz, together with BPFI, BSF and FTF). The dominant energy sits at BPFO (amplitude about 753.9 against about 280.3 for BPFI and BSF), and about 98.9% of the fault windows are classified as outer race faults, which agrees with the known failure of this bearing.

**RUL target.** Failure is defined as the point where the normalised HI reaches 0.85, and `RUL_true` is the time remaining until that point for every window.

**Models.** The following regressors were trained on an 80/20 split of the 402 windows and evaluated with MAE, RMSE and R²: PCA + Random Forest, Kernel PCA + Random Forest, Autoencoder (27 → 64 → 32 → 8 latent) + Random Forest, PCA + XGBoost, PCA + Extra Trees, PCA + SVR and a two layer LSTM on sequences of 20 windows. `oelp_RUL_AllModels.ipynb` additionally includes PCA + Gradient Boosting and PCA + Ridge.

## Results

The figures below are the ones reported in the final report and presentation.

| Model | R² | MAE (s) | RMSE (s) |
|---|---|---|---|
| Autoencoder + Random Forest | 0.9616 | 42.7 | 60.8 |
| PCA + Extra Trees | 0.9449 | | |
| PCA + XGBoost | 0.9347 | | |
| PCA + Random Forest | 0.9317 | | |
| Kernel PCA + Random Forest | 0.8566 | | |
| PCA + SVR | 0.4474 | | |
| LSTM | −3.5668 | | |

The saved outputs in the notebooks come from later runs and differ from these numbers. `oelp_part2_final.ipynb` shows Autoencoder + RF at R² 0.9716 (MAE 36.4 s, RMSE 52.3 s), and `oelp_RUL_AllModels.ipynb` ranks PCA + Gradient Boosting first at R² 0.9811. The variation comes mainly from the autoencoder and LSTM not being seeded. Please read the limitations section before relying on any of these figures.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── OELP Final Report.docx            final written report
├── P157 OELP Presentation.pptx       final presentation
├── Datasets/
│   └── rul_features_final_dataset.csv
├── Literature/                       reference material
├── Monthly reports/                  monthly progress reports 1 to 4
└── Work progress/
    ├── OELP-2 notes.docx
    ├── oelp slides-1.pptx
    ├── final code files/
    │   ├── oelp_overview_final.ipynb     full pipeline, end to end, with all plots
    │   ├── oelp_part1_easy_review.ipynb  condensed pipeline that writes the feature CSV
    │   ├── oelp_part2_final.ipynb        model training and comparison (7 models)
    │   ├── oelp_RUL_AllModels.ipynb      extended comparison (9 models)
    │   └── features_extraction.ipynb     early feature exploration on a separate 5 kHz signal
    └── extra code files/
        ├── oelp_part1_nasa.ipynb         exploratory study of NASA IMS
        ├── oelp_part1_PRONOSTIA.ipynb    exploratory study of PRONOSTIA
        ├── oelp_part1_xjtu-sy.ipynb      exploratory study of XJTU‑SY
        └── RUL_NASA_Updated.ipynb        earlier version of the NASA pipeline
```

## How to run

Create an environment with Python 3.10 or newer and install the dependencies:

```bash
pip install -r requirements.txt
```

Run `Work progress/final code files/oelp_part1_easy_review.ipynb` first. It downloads (or reads) the IMS Test 2 data, extracts the features, builds the HI and RUL target, and writes `rul_features_final_dataset.csv` and `rul_top_features_dataset.csv` into the working directory. Then run `oelp_part2_final.ipynb` or `oelp_RUL_AllModels.ipynb` from the same directory to train and compare the models. If you only want to reproduce the modelling step, copy `Datasets/rul_features_final_dataset.csv` next to the part 2 notebook, since the notebooks read it from the current directory.

`oelp_overview_final.ipynb` repeats the whole pipeline in one notebook with every intermediate plot, and is the best place to start reading.

## Limitations and future work

The time axis used throughout is the concatenated recording time, not wall clock time. Because each 1 s snapshot was taken 10 minutes apart, a 5 s window actually spans roughly 50 minutes of machine operation and the full 1,007 s of signal covers about 164 hours. RUL values in seconds should therefore be read as positions along the recording rather than as real remaining life, and a version expressed in hours would be more meaningful.

The models were evaluated with a random 80/20 split of overlapping windows from a single bearing. Neighbouring windows share half of their samples, so most test windows have an almost identical twin in the training set, and the scores mainly measure interpolation within one run. In addition, `RUL_true` in the shipped CSV equals 997.5 minus `time` for every window, so the regression target is a straight line in time. A chronological split, or training on one bearing and testing on another, is needed before the results can be taken as a measure of prognostic accuracy. Natural next steps are to validate on the other IMS failures (Test 1 Bearings 3 and 4, Test 3 Bearing 3) and on the PRONOSTIA and XJTU‑SY bearings that were already explored, to report prediction intervals rather than point estimates, and to add PHM specific metrics such as the PHM 2012 scoring function.

## References

1. J. Lee, H. Qiu, G. Yu, J. Lin. "Bearing Data Set." IMS, University of Cincinnati. NASA Ames Prognostics Data Repository, 2007.
2. P. Nectoux et al. "PRONOSTIA: An Experimental Platform for Bearings Accelerated Degradation Tests." IEEE ICPHM, 2012.
3. B. Wang et al. "XJTU‑SY Rolling Element Bearing Accelerated Life Test Dataset." Xi'an Jiaotong University, 2018.
4. N. Gebraeel et al. "Residual Life Predictions from Vibration Based Degradation Signals: A Neural Network Approach." IEEE Transactions on Industrial Electronics, 2004.
5. Y. Lei, N. Li, L. Guo. "Machinery Health Prognostics: A Systematic Review from Data Acquisition to RUL Prediction." Mechanical Systems and Signal Processing, 2018.
6. C. Chen et al. "Machine Remaining Useful Life Prediction via an Attention Based Deep Learning Approach." IEEE Transactions on Industrial Electronics, 2021.

The full reference list is in `OELP Final Report.docx`.
