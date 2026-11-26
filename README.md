# Turbofan Engine Remaining Useful Life (RUL) Prediction

A PyTorch deep learning pipeline for aircraft engine predictive maintenance using the **NASA C-MAPSS (FD001)** turbofan degradation dataset. This project implements, trains, and benchmarks two sequence-modeling architectures—a stacked **LSTM** baseline and a **Time-Series Transformer**—to forecast Remaining Useful Life (RUL) from multivariate sensor telemetry.

---

## Table of Contents

- [Overview](#overview)
- [Performance Results](#performance-results)
- [Dataset Overview](#dataset-overview)
- [Data Engineering Pipeline](#data-engineering-pipeline)
- [Model Architectures](#model-architectures)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

---

## Overview

Unplanned aircraft engine failures are costly and safety-critical. Using run-to-failure multivariate time-series telemetry from NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), this project predicts how many flight cycles remain before an engine requires maintenance.

### Key Highlights
- **End-to-End PyTorch Workflow:** Covers raw data ingestion, piecewise RUL labeling, `MinMaxScaler` normalization, sliding-window tensor generation, GPU-accelerated training, comparative evaluation, and single-engine inference.
- **LSTM vs. Transformer Benchmark:** Compares recurrent sequential modeling against multi-head self-attention on identical 30-cycle sliding windows.
- **Stable Convergence via Target Scaling:** Normalizes RUL targets to `[0, 1]` during training (scaled by the 125-cycle cap) to prevent mean-prediction flatlining, then un-scales predictions back to flight cycles at evaluation and inference time.

---

## Performance Results

Both architectures were trained for **50 epochs** on normalized `FD001` telemetry (`batch_size=64`) and evaluated on the final 30-cycle window of all **100 unseen test engines**.

| Architecture | Learning Rate | Gradient Clipping | Test RMSE (Flight Cycles) | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer (`RUL_Transformer`)** | `0.0005` | `max_norm=1.0` | **14.59** | **Best Model** |
| **LSTM (`RUL_LSTM`)** | `0.001` | None | **14.74** | **Baseline** |

### Visual Comparison

![LSTM vs. Transformer RUL Prediction Comparison](models/final_evaluation_graph.png)

### Key Takeaways
- The **Transformer** outperformed the stacked **LSTM** on unseen test engines (14.59 vs. 14.74 RMSE cycles) by modeling global relationships across all 30 cycles in parallel via self-attention.
- Both models track engine degradation most closely in the critical late-life regime (`RUL < 50
