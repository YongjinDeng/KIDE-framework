# Robust Markerless Surface-to-Internal Tracking via Kinematics-Informed Prior (KIDE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB 2021+](https://img.shields.io/badge/MATLAB-2021+-blue.svg)]()

This repository contains the official implementation of the **KIDE** framework, as described in our paper submitted to *Artificial Intelligence in Medicine*.

## 📌 Overview
Unconstrained data-driven models (e.g., OLS, unconstrained deep learning) are highly vulnerable to clinical occlusion noise in Surface-Guided Radiotherapy (SGRT). **KIDE** introduces a patient-specific kinematic spectral penalty that acts as a biomechanical low-pass filter. It mathematically guarantees 0.00% Jacobian folding (microscopic topology) and effectively prevents false dosimetric alarms (macroscopic geometry).

## 🚀 Key Features
- **Geometric Robustness:** Maintains high clinical tracking success rates even under 15% structural occlusion noise.
- **Biomechanical Plausibility:** 100% physically plausible deformation vector fields (0% Jacobian folding).
- **Heterogeneity-Aware Dosimetry:** Built-in pseudo-dose engine utilizing Radiological Path Lengths (RPL) for D95% and V20 evaluation.

## 📂 Quick Start
1. Clone the repository: `git clone https://github.com/YourName/KIDE-framework.git`
2. Download the publicly available [DIR-Lab 4DCT dataset](http://www.dir-lab.com/).
3. Run `src/Phase1_Geometric_Evaluation.m` to reproduce TRE boxplots and Jacobian analysis.
4. Run `src/Phase2_Dosimetric_Evaluation.m` to generate dose maps and Dose-Volume Histograms (DVH).

## 📊 Citation
If you find this code useful for your research, please consider citing our paper (Citation details to be updated upon publication).
