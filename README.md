# G2Net Gravitational Wave Detection

[Link to Kaggle Training Notebook](INSERISCI_QUI_IL_TUO_LINK_KAGGLE)

This repository contains a Deep Learning pipeline designed to detect gravitational wave signals from binary black hole collisions within noisy interferometric data (LIGO/Virgo). The solution utilizes a **Constant Q-Transform (CQT)** for time-frequency conversion and an **EfficientNet-B2** CNN backbone for binary classification.

## Training Context & Hardware

Due to the high computational cost of CQT spectrogram generation and Convolutional Neural Network training, the project workflow is divided as follows:

* **Primary Training:** Performed on **Kaggle** using a **NVIDIA Tesla P100 GPU**.
* **Training Duration:** Approximately 4 hours for 5 folds, 8 Epochs each on the full dataset.
* **Local Repository:** Designed for **reproducibility, inference, and visualization**. It allows verifying the code logic and generating scientific plots on a standard CPU without repeating the computationally expensive training phase.

The full training history (Loss and AUC metrics) and model weights were exported from the GPU session to be analyzed locally.

## Project Structure

The project is organized into modular components:

```text
.
├── data/                     # Raw and processed data storage
├── outputs/                  # Generated logs, models, and visualization plots
├── scripts/                  # Executable pipeline scripts
│   ├── debug_check.py        # Verifies system integrity and data paths
│   ├── train.py              # Local Training Loop (Demonstration/Proof of Concept)
│   ├── import_kaggle_logs.py # Imports pre-computed GPU logs for analysis
│   └── visualize.py          # Generates scientific plots (Spectrograms, ROC, Grid)
├── src/                      # Core source code (Configuration, Dataset, Model)
├── pyproject.toml            # Dependency management (uv)
└── README.md                 # Project documentation
````

## Installation & Requirements

This project uses **uv** for Python dependency management (Python 3.11).

1. **Install uv** (if not already installed):
    
    ```Bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```
    
1. **Initialize Environment:** Run the following command in the project root to install dependencies (PyTorch, nnAudio, Timm, etc.):
    
    ```Bash
    uv sync
    ```
    

## Usage Guide (Reproducibility Workflow)

Follow these steps to verify the pipeline functionality and reproduce the results.

### 1. System Check

Verify that the data paths and hardware are correctly detected by the environment.

```Bash
uv run scripts/debug_check.py
```

### 2. Training Demo (Proof of Concept)

Launch the training script locally to verify that the code executes correctly, the data loader functions, and the loss is calculated.

```Bash
uv run scripts/train.py
```

**Note:** This script runs on the local CPU. It is functionally identical to the Kaggle training but significantly slower. It is recommended to stop the process (using `CTRL + C`) once the training progress bar appears and the first loss metrics are printed, as this confirms the validity of the code.

### 3. Import Full Training Logs

Import the official loss and AUC history recorded during the full GPU training session. This allows for the analysis of the complete training dynamics without the need for local re-training.

```Bash
uv run scripts/import_kaggle_logs.py
```

### 4. Visualization & Results

Generate high-definition scientific plots using the processed data and imported logs.

```Bash
uv run scripts/visualize.py
```

Output files will be saved in `outputs/plots/`:

- **Spectrogram Reconstruction:** Visual representation of Gravitational Waves (CQT).
    
- **Training Grid:** Loss and AUC curves across all folds.
    
- **Confusion Matrix & ROC Curve:** Global classification performance metrics.
    

## Methodology

1. **Preprocessing:** Raw time-series waves from LIGO Hanford, LIGO Livingston, and Virgo are converted into images using **CQT (Constant Q-Transform)**. This transforms the 1D signal processing problem into a 2D computer vision task.
    
2. **Model Architecture:** An **EfficientNet-B2** (pre-trained on ImageNet) is used as the backbone to extract features from the spectrograms.
    
3. **Training Strategy:** The model is trained using Binary Cross Entropy (BCE) loss with a Stratified K-Fold Cross-Validation strategy (5 Folds) to ensure robust generalization.
    

## Results

The model demonstrates strong capability in distinguishing signal from noise, achieving competitive AUC scores on the validation set. The spectral whitening and CQT preprocessing steps are critical for isolating the "chirp" signal characteristic of binary black hole mergers.

---

**Authors:** Stefano Solazzo, Tina Rabbani, Daroui Dong
