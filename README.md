# Real-Time Speech Enhancement Implementation on FPGA using SNN

This is the repository for the final project of **Emre Yılmaz** and **Erdem Sezen**. It contains an Audio Denoising Spiking Neural Network (SNN) implementation aimed at real-time speech enhancement on FPGA platforms. The primary objective is to design an SNN that performs well under limited hardware resources.

---

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- torchaudio
- matplotlib
- tqdm

## Setup

### 1. Install Spiker+

This repository depends on Spiker+, included in the spiker/ directory. Install it using:

    cd spiker
    pip install .

Or equivalently:

    cd spiker
    python setup.py install

### 2. Prepare Dataset

Place your dataset into the following folder structure:

    audio/
    ├── clean/ 
    └── noisy/  

### 3. Configuration

You can edit training and model parameters in:

    utils/config.py

## Usage

### Train the Model

    python train.py

### Evaluate the Model

    python evaluate.py

### Visualize Results

    python visualize.py

## Repository Structure

    ├── audio/                
    │   ├── clean/
    │   └── noisy/
    ├── data/
    ├── models/
    ├── outputs/
    ├── spiker/ 
    ├── Trained/               
    ├── utils/                
    ├── train.py               
    ├── evaluate.py             
    ├── visualize.py            