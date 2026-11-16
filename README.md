# A Robust Framework for the Classification of Steady-State Visually Evoked Potentials Using a Multi-Level Residual Network

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Institution: CNR-ICAR](https://img.shields.io/badge/Institution-CNR--ICAR-darkblue.svg)](https://www.icar.cnr.it/)

Official research repository containing the source code, experimental pipelines, and dataset processing workflows for the paper:

> **"A Robust Framework for the Classification of Steady-State Visually Evoked Potentials Using a Multi-Level Residual Network"**

---

## Authors & Affiliation

- **Francesco Capria**<sup>1</sup> [[ORCID](https://orcid.org/0009-0000-5779-9317)] &bull; `francesco.capria@icar.cnr.it`
- **Franco Cicirelli**<sup>1</sup> [[ORCID](https://orcid.org/0000-0002-6138-5739)] &bull; `franco.cicirelli@icar.cnr.it`
- **Alberto Falcone**<sup>1,&lowast;</sup> [[ORCID](https://orcid.org/0000-0002-2660-1432)] &bull; `alberto.falcone@icar.cnr.it`
- **Massimo Guarascio**<sup>1</sup> [[ORCID](https://orcid.org/0000-0001-7711-9833)] &bull; `massimo.guarascio@icar.cnr.it`
- **Antonio Guerrieri**<sup>1</sup> [[ORCID](https://orcid.org/0000-0003-1469-9484)] &bull; `antonio.guerrieri@icar.cnr.it`
- **Luigi Rizzo**<sup>1</sup> [[ORCID](https://orcid.org/0009-0003-5031-5668)] &bull; `luigi.rizzo@icar.cnr.it`

<sup>1</sup> **National Research Council of Italy, Institute for High Performance Computing and Networking (CNR-ICAR)**  
Via P. Bucci 8/9C, 87036 Rende (CS), Italy  
<sup>&lowast;</sup> *Corresponding Author*: Alberto Falcone (`alberto.falcone@icar.cnr.it`)

---

## Overview

Steady-State Visually Evoked Potentials (SSVEP / SVEP) are widely utilized in Brain-Computer Interfaces (BCIs) due to their high signal-to-noise ratio and information transfer rates. However, real-world deployment faces major challenges, including inter-subject variability, data scarcity, real-time computational constraints, and non-stationarity of electroencephalogram (EEG) signals.

This framework introduces a robust end-to-end BCI pipeline centered on a novel **Multi-Level Residual Network (ML-ResNet)** architecture:
- **EEG Signal Conditioning & PSD Feature Extraction:** Temporal segmentation with sliding windows and Welch-based Power Spectral Density (PSD) analysis targeting occipital and parieto-occipital channels.
- **Multi-Level Residual Architecture:** Residual blocks combined with multi-scale skip concatenation that preserve low-level and high-level spectral features.
- **Extensive Validation:**
  - Benchmark against traditional and deep classifiers (Random Forest, SVM with RBF kernel, and MLP).
  - Scalability across increasing subject cohorts (13, 15, 20, 27 subjects).
  - Robustness under severe training data scarcity.
  - Computational latency & throughput analysis for real-time BCI operation.
  - Cross-subject transfer learning and domain adaptation strategies on unseen subjects.

---

## System in Action

Demonstration of the SVEP-based BCI framework during operation:

<div align="center">
  <video src="assets/SVEP-based-BCI-Framework-in_action.mp4" width="100%" controls>
    <p>Your browser does not support the video tag. You can view or download the video directly <a href="assets/SVEP-based-BCI-Framework-in_action.mp4">here</a>.</p>
  </video>
</div>

> 🎬 **Direct Link:** [`assets/SVEP-based-BCI-Framework-in_action.mp4`](assets/SVEP-based-BCI-Framework-in_action.mp4)

---

## Repository Structure

```text
SVEP-based-BCI-Framework/
├── 00-Preprocessing_Dataset/
│   ├── Test_0.ipynb                     # Raw EEG filtering, windowing & PSD feature extraction
│   └── README.txt                       # Preprocessing pipeline notes
├── 01-Classifier_Selection/
│   ├── Classifier_Selection.ipynb       # Benchmark: RF, SVM, MLP vs. Multi-Level ResNet
│   └── README.txt                       # Classifier selection & evaluation documentation
├── 02-Test_1_Multi_Subject_Training/
│   ├── Muli_Sub_ResNet.ipynb            # Multi-subject scaling analysis (13, 15, 20, 27 subjects)
│   └── README.txt                       # Multi-subject experimental setup
├── 03-Test_2_Data_Scarcity/
│   ├── Data_Scarcity_ResNet.ipynb       # Performance evaluation across incremental training folds (P1-P10)
│   └── README.txt                       # Data scarcity protocol details
├── 04-Test_3_Scalability_Analysis/
│   └── Scalability_Analysis.ipynb       # Computational training runtime and one-shot inference timing
├── 05-Test_4_Cross_Subject_Adaptation/
│   ├── Test_4_1.ipynb                   # Adaptation with frozen backbone (softmax fine-tuning only)
│   ├── Test_4_2.ipynb                   # Adaptation with full Dense layer unfreezing
│   └── README.txt                       # Domain adaptation & transfer learning methodology
├── assets/                              # Documentation media, figures & demo recordings
│   └── SVEP-based-BCI-Framework-in_action.mp4 # Video demonstration of the BCI framework in action
├── Raw_Data/                            # Raw EEG recordings from 27 subjects (Bitbrain Diadem)
│   ├── EEG_Subject_1.csv ... EEG_Subject_27.csv
├── PSD_data/                            # Preprocessed PSD feature matrices for all 27 subjects
│   ├── PSD_EEG_Subject_1.csv ... PSD_EEG_Subject_27.csv
├── requirements.txt                     # Python dependencies
├── LICENSE                              # CC0 1.0 Universal License
└── README.md                            # Main documentation
```

---

## Dataset & Acquisition Hardware

- **Device:** [Bitbrain Diadem](https://www.bitbrain.com/) wireless EEG headset.
- **Sampling Frequency ($f_s$):** 256 Hz.
- **Subjects:** 27 healthy participants.
- **Selected Electrodes:** Occipital and parieto-occipital channels:
  - `PO7` (Channel 9)
  - `O1` (Channel 10)
  - `O2` (Channel 11)
  - `PO8` (Channel 12)
- **Visual Stimulus Classes & Event Markers:**
  - **Class 0 / Rest:** Standby condition
  - **Class 33024:** No stimulus
  - **Class 33025:** 10.00 Hz visual flicker
  - **Class 33026:** 12.00 Hz visual flicker
  - **Class 33027:** 8.57 Hz visual flicker

---

## Methodological Pipeline

```
Raw EEG Signals (256 Hz)
       │
       ▼
Signal Preprocessing
   ├── Bandpass Filter: 4th-order Butterworth (6 – 30 Hz)
   └── Notch Filter: IIR notch at 50 Hz (Q = 30)
       │
       ▼
Temporal Windowing
   ├── Window Length: W = 512 samples (2.0 s)
   └── Window Step: S = 64 samples (0.25 s, 75% overlap)
       │
       ▼
Spectral Feature Extraction
   └── Welch PSD (nperseg=256, noverlap=192)
       └── 50 frequency bins × 4 channels = 200 features
       │
       ▼
Data Normalization & Balancing
   ├── MinMaxScaler (strictly fitted on training splits to prevent data leakage)
   └── Minority class oversampling (applied strictly to training data)
       │
       ▼
Multi-Level Residual Network (ML-ResNet)
       │
       ▼
Classification & Real-Time BCI Output
```

### Multi-Level Residual Network (ML-ResNet) Architecture

The network is engineered to handle non-stationary EEG features through hierarchical residual connections and multi-level feature aggregation:
1. **Input Layer:** Receives the 200-dimensional PSD feature vector.
2. **Dense Base Layer:** 256 units with ReLU activation, Batch Normalization, and Dropout ($p = 0.25$).
3. **Multi-Stage Residual Blocks ($K = 5$ blocks):**
   - Each block incorporates Dense projection, Batch Normalization, Dropout, and an identity/shortcut addition:
     $$\mathbf{y}_k = \mathbf{x}_k + \mathcal{F}(\mathbf{x}_k, \mathcal{W}_k)$$
   - Followed by a secondary transformation block.
   - Intermediate outputs from all $K$ levels are preserved in a multi-level stack.
4. **Multi-Level Skip Concatenation:**
   - Aggregates the base representation, all 5 intermediate block representations, and the original input vector:
     $$\mathbf{z} = \operatorname{Concat}\left(\mathbf{h}_{\mathrm{base}}, \mathbf{h}_1, \dots, \mathbf{h}_K, \mathbf{x}_{\mathrm{inp}}\right)$$
5. **Dense Transition & Output:**
   - Dense layer (256 units, ReLU, BatchNorm, Dropout).
   - Dense classification layer with Softmax activation across the target stimulus classes.
6. **Training Configuration:**
   - Optimizer: RMSprop (learning rate = $10^{-3}$).
   - Loss function: Categorical Cross-Entropy.
   - Learning rate schedule: `ReduceLROnPlateau` (`factor = 0.5`, `patience = 5`, `min_lr = 1e-6`).
   - Checkpointing: Best model weights saved based on validation loss.

---

## Experimental Framework

The experiments are organized into reproducible Jupyter notebooks:

### 1. Preprocessing (`00-Preprocessing_Dataset`)
- Implements bandpass and notch filtering.
- Performs temporal segmentation with a sliding window approach.
- Computes Welch PSD features per channel and generates frequency-domain representations for all 27 subjects.

### 2. Classifier Selection & Baseline Comparison (`01-Classifier_Selection`)
- Compares 4 classification paradigms under identical, leakage-free cross-validation:
  1. **Random Forest (RF):** Classic non-linear ensemble.
  2. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel.
  3. **Multi-Layer Perceptron (MLP):** Three-stage fully connected network ($256 \to 128 \to 64$).
  4. **Multi-Level Residual Network (ML-ResNet):** Proposed architecture.
- Evaluates Accuracy, Precision, Recall, and F1-score across all subjects.

### 3. Subject Cohort Scaling (`02-Test_1_Multi_Subject_Training`)
- Assesses generalization when aggregating training data across increasing participant cohorts:
  - Evaluated on merged pools of **13**, **15**, **20**, and **27** subjects.
  - Automatically exports merged datasets, metrics, and trained weights for auditability.

### 4. Training Data Scarcity Evaluation (`03-Test_2_Data_Scarcity`)
- Investigates classifier resilience when limited calibration data is available:
  - Fixes a 50% test partition across all 27 subjects.
  - Subdivides the training partition into 10 disjoint, stratified subsets ($P_1, \dots, P_{10}$).
  - Performs incremental training across $k \in \{1, \dots, 10\}$ partitions to plot learning curves under scarcity.

### 5. Scalability & Real-Time Latency Analysis (`04-Test_3_Scalability_Analysis`)
- Measures computational training overhead across dataset scales.
- Profiles single-sample one-shot inference latency (including GPU warm-up and timing).
- Validates that inference latency satisfies the real-time threshold imposed by the 250 ms window step ($S = 64$ samples at 256 Hz).

### 6. Cross-Subject Adaptation & Transfer Learning (`05-Test_4_Cross_Subject_Adaptation`)
- Evaluates model transferability to completely unseen subjects:
  - **Source Model:** Pre-trained on 15 subjects.
  - **Target Cohort:** 12 independent, unseen subjects.
  - **Test 4.1 (Linear Head Adaptation):** Backbone weights frozen; only the final softmax classification layer is adapted.
  - **Test 4.2 (Multi-Level Fine-Tuning):** Full Dense layers (base, residual blocks, and classifier) unfrozen for domain adaptation.
  - Demonstrates substantial performance recovery on new subjects with minimal calibration data.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Recommended: Dedicated virtual environment (`venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/massimo-guarascio/SVEP-based-BCI-Framework.git
   cd SVEP-based-BCI-Framework
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Experiments

Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
```

Execute the notebooks in sequence or navigate directly to the experiment of interest:
1. `00-Preprocessing_Dataset/Test_0.ipynb` (To regenerate PSD features from raw data)
2. `01-Classifier_Selection/Classifier_Selection.ipynb`
3. `02-Test_1_Multi_Subject_Training/Muli_Sub_ResNet.ipynb`
4. `03-Test_2_Data_Scarcity/Data_Scarcity_ResNet.ipynb`
5. `04-Test_3_Scalability_Analysis/Scalability_Analysis.ipynb`
6. `05-Test_4_Cross_Subject_Adaptation/Test_4_1.ipynb` and `Test_4_2.ipynb`

---

## Citation

If you use this codebase, framework, or dataset in your research, please cite our paper:

### APA Format
> Capria, F., Cicirelli, F., Falcone, A., Guarascio, M., Guerrieri, A., & Rizzo, L. *A Robust Framework for the Classification of Steady-State Visually Evoked Potentials Using a Multi-Level Residual Network*. Internet of Things, 2026. Elsevier.

### BibTeX
```bibtex
@article{capria2024robust,
  title     = {A Robust Framework for the Classification of Steady-State Visually Evoked Potentials Using a Multi-Level Residual Network},
  author    = {Capria, Francesco and Cicirelli, Franco and Falcone, Alberto and Guarascio, Massimo and Guerrieri, Antonio and Rizzo, Luigi},
  journal   = {Internet of Things - Under Review},
  year      = {2026},
  publisher = {Elsevier}
}
```

---

## License

This project and associated materials are licensed under the **Creative Commons CC0 1.0 Universal** (CC0 1.0) Public Domain Dedication. See the [LICENSE](LICENSE) file for full details.

---

## Contact & Acknowledgements

For questions, collaborations, or issues regarding this framework, please contact:
- **Alberto Falcone** &bull; `alberto.falcone@icar.cnr.it`
- **National Research Council of Italy (CNR-ICAR)** &bull; [www.icar.cnr.it](https://www.icar.cnr.it/)