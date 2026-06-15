# Physics-Informed Depth-Enhanced (PIDE) Framework

## Overview
This repository contains the MATLAB implementation of the Physics-Informed Depth-Enhanced (PIDE) framework for surface-to-internal deformation prediction in lung SBRT.

## Publication
Deng Y, Chen R. Physics-informed surface-to-internal deformation prediction for intra-fraction motion management in lung SBRT: An in silico geometric and dosimetric validation. *Radiotherapy and Oncology*, 2026.

## Requirements
- MATLAB R2020a or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

## Data
This code uses the publicly available DIR-Lab 4DCT dataset.  
Access: http://www.dir-lab.com/

## Usage
1. Download DIR-Lab data to `./data/Case1Pack/` ... `./data/Case10Pack/`
2. Run `Batch_Process_All_Cases.m`
3. Run `Phase1_Batch_Process_Baselines.m`
4. Run `Phase2_Clinical_Dose_Engine.m`

## Citation
If you use this code, please cite our paper.
