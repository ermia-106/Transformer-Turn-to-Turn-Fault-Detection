# ⚡ Transformer Turn-to-Turn Fault Detection

A machine learning project for detecting **turn-to-turn short-circuit faults in transformer windings** using electrical current signals generated from **MATLAB/Simulink**, engineered signal features, and an **XGBoost classifier**.

The project covers the complete workflow from transformer simulation and feature extraction to machine learning classification, SHAP-based explainability, and an interactive **Streamlit inference dashboard**.

---

## 📌 Overview

Turn-to-turn faults are important internal transformer faults that can develop rapidly and may cause severe damage if they are not detected in time.

This project uses simulated three-phase transformer current signals to extract:

* 📊 Statistical features
* ⚡ Electrical and symmetrical-component features
* 🌊 Harmonic features
* 📈 Wavelet-domain features

These features are then used by an **XGBoost binary classification model** to distinguish between:

* 🟢 **Healthy Transformer**
* 🔴 **Turn-to-Turn Fault**

### 🔄 Complete Workflow

The project consists of two main pipelines.

#### 🏗️ Training Pipeline

1. ⚙️ Generate transformer operating scenarios in MATLAB/Simulink.
2. 🔌 Simulate healthy and faulty transformer conditions.
3. 🔄 Resample generated signals to a fixed sampling interval.
4. 🧮 Extract 49 signal-based features.
5. ➕ Add `Load_Power` as the 50th model input feature.
6. 🤖 Train and evaluate an XGBoost classifier.
7. 🔍 Perform SHAP explainability analysis.
8. 💾 Save the trained model for later inference.

#### 🚀 Inference Pipeline

1. 📥 Receive three-phase current signals (`Ia`, `Ib`, `Ic`).
2. ✅ Validate the input signals.
3. 🧮 Extract the same 49 features used during training.
4. ➕ Add `Load_Power`.
5. 🤖 Load the trained XGBoost model.
6. 🔮 Predict the transformer condition.
7. 📊 Calculate class probabilities.
8. 🖥️ Display the results through a Streamlit dashboard.

---

# 📂 Project Structure

```text
Transformer-Turn-to-Turn-Fault-Detection/
│
├── Heatmap/
│   ├── Confusion_Matrix_Heatmap.png
│   └── Correlation_Heatmap.png
│
├── Matlab/
│   ├── Generate_Transformer_Dataset/
│   │   ├── Generate_Transformer_Dataset.m
│   │   └── ML_Transformer_FixedStep_2kHz.csv
│   │
│   ├── Extract_Transformer_Features/
│   │   ├── Extract_Transformer_Features.m
│   │   └── ML_Features_All_49_new6.csv
│   │
│   └── Simulink_Transformer_Model/
│       ├── Simulink_Transformer_Model.slx
│       └── Simulink_Transformer_Model_Circuit.png
│
├── Models/
│   ├── SHAP_Results.pkl
│   ├── XGBoost_Model.pkl
│   └── Simulink_Transformer_Model.slx
│
├── SHAP Analysis/
│   ├── SHAP_Dependence.png
│   ├── SHAP_Force.png
│   ├── SHAP_Summary.png
│   └── SHAP_Top_15_Feature_Importance.png
│
├── Streamlit/
│   ├── Feature_Extraction_Inference.py
│   ├── Inference.py
│   └── app.py
│
├── Matlab_Extracted_ML_Features.csv
├── Transformer-Turn-to-Turn-Fault-Detection.py
├── Transformer-Turn-to-Turn-Fault-Detection.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

# ⚙️ Dataset Generation

The dataset is generated using **MATLAB and Simulink** rather than being downloaded from an external dataset repository.

The MATLAB generation script simulates healthy and faulty transformer scenarios with different:

* ⚡ Load powers
* 🔥 Fault percentages
* ⏱️ Fault initiation times

The simulation uses a variable-step solver and then resamples the generated signals to a fixed sampling interval of:

```text
Sampling interval: 0.0005 seconds
Sampling frequency: 2 kHz
```

### 📊 Generated Dataset

The raw dataset contains:

| Feature         | Description                    |
| --------------- | ------------------------------ |
| `Time`          | Simulation time                |
| `Ia`            | Phase-A current                |
| `Ib`            | Phase-B current                |
| `Ic`            | Phase-C current                |
| `Load_Power`    | Transformer load power         |
| `Fault_Percent` | Fault percentage               |
| `Scenario_ID`   | Simulation scenario identifier |
| `Label`         | Healthy/Fault class            |

---

# 🧮 Feature Extraction

The MATLAB feature-extraction script creates **49 signal-based features** from each current-signal window.

The final machine-learning input contains:

```text
49 signal-derived features
+
Load_Power
=
50 model input features
```

### 🚫 Features Not Used by the Model

The following columns are not used as model inputs:

```text
Fault_Percent
Scenario_ID
Label
```

`Scenario_ID` is only used for grouped train/test splitting and cross-validation.

---

## 🪟 Window Configuration

The feature extraction uses:

```text
Window size       : 80 samples
Step size         : 40 samples
Sampling frequency: 2 kHz
Window duration   : 40 ms
Wavelet           : db4
Decomposition     : Level 5
```

At a sampling frequency of 2 kHz:

```text
80 samples × 0.0005 s = 0.04 s = 40 ms
```

Therefore, each feature window contains approximately **two cycles of a 50 Hz electrical signal**.

---

# 📊 Time-Domain Features

The first 19 features are calculated from the three-phase current signals.

These include:

* 📏 RMS
* 📈 Peak value
* 📐 Standard deviation
* 📊 Crest factor
* 🌊 Total harmonic distortion (THD)
* ⚡ Zero-sequence component
* ⚡ Positive-sequence component
* ⚡ Negative-sequence component
* ⚖️ Current unbalance

The symmetrical components are calculated from the fundamental-frequency components of the three phase currents.

---

# 🌊 Wavelet-Domain Features

Wavelet decomposition is performed using the **Daubechies 4 (****`db4`****)** mother wavelet with decomposition level 5.

The extracted wavelet features are:

### ⚡ Wavelet Energy

Energy is calculated from D1–D5 for all three phases.

```text
5 levels × 3 phases = 15 features
```

### 🧠 Wavelet Entropy

Entropy is calculated for the detail coefficients of Phase A.

```text
5 features
```

### 📉 Wavelet Variance

Variance is calculated for the Phase-A detail coefficients.

```text
5 features
```

### 📐 Wavelet Standard Deviation

Standard deviation is calculated for the Phase-A detail coefficients.

```text
5 features
```

### 📦 Feature Summary

```text
19 Time-domain / Electrical features
15 Wavelet-energy features
 5 Wavelet-entropy features
 5 Wavelet-variance features
 5 Wavelet-standard-deviation features
---------------------------------------
49 Signal features
```

---

# 🤖 Machine Learning Model

The classification model is based on **XGBoost**.

The model receives **50 input features**:

```text
49 extracted signal features
+
Load_Power
```

### ⚙️ XGBoost Configuration

```python
XGBClassifier(
    n_estimators=106,
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    reg_alpha=0.1,
    reg_lambda=3,
    random_state=42,
    eval_metric='logloss'
)
```

---

# 🛡️ Data Leakage Prevention

One of the important considerations in this project is **data leakage**.

Multiple feature windows can originate from the same transformer simulation scenario. Therefore, a normal random row-wise train/test split could place highly similar windows from the same scenario into both training and testing datasets.

To reduce this problem, `Scenario_ID` is used as a grouping variable.

### 🔀 Train/Test Split

The project uses:

```text
GroupShuffleSplit
```

with:

```text
Train size: 81%
Test size : 19%
Random state: 42
```

### 🔁 Cross-Validation

Model validation uses:

```text
10-Fold GroupKFold
```

This ensures that samples belonging to the same simulation scenario are kept within the same fold.

`Scenario_ID` is removed before training and is never provided to the XGBoost model.

---

# 📈 Model Evaluation

## 🧪 Test Set

The latest test-set results are:

```text
Accuracy  : 81.23%
Precision : 80.54%
Recall    : 91.98%
F1 Score  : 85.88%
```

### 🔁 10-Fold GroupKFold

```text
Accuracy  : 0.77 ± 0.02
Precision : 0.79 ± 0.02
Recall    : 0.86 ± 0.04
F1 Score  : 0.82 ± 0.02
```

The grouped cross-validation provides a more realistic evaluation because the model is evaluated on previously unseen simulation scenarios.

---

# 🔍 SHAP Explainability

**SHAP (SHapley Additive exPlanations)** is used to analyze the contribution of individual features to model predictions.

The project generates the following outputs:

```text
SHAP_Top_15_Feature_Importance.png
SHAP_Summary.png
SHAP_Dependence.png
SHAP_Force.png
SHAP_Results.pkl
```

These visualizations provide insight into which electrical and wavelet features have the greatest influence on the classifier's predictions.

---

# 💾 Saved Model

The trained XGBoost model is serialized and saved as:

```text
Models/XGBoost_Model.pkl
```

This model is used directly by the inference pipeline.

---

# 🚀 Inference Pipeline

A separate inference pipeline was developed for deployment.

The main goal is to keep the original training and MATLAB workflows unchanged while providing a lightweight Python-based inference system.

### 🔄 Inference Flow

```text
Ia ─┐
Ib ─┼──► Python Feature Extraction ──► 49 Features
Ic ─┘                                      │
                                           ▼
                                      Load_Power
                                           │
                                           ▼
                                      50 Features
                                           │
                                           ▼
                                   XGBoost Model
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                           Healthy                    Fault
```

---

# 🐍 Python Feature Extraction

The file:

```text
Feature_Extraction_Inference.py
```

reproduces the original MATLAB feature-extraction logic for inference.

It generates the same **49 features in the same order** as the training pipeline.

The implementation includes:

* 📏 RMS
* 📈 Peak
* 📐 Standard deviation
* 📊 Crest factor
* 🌊 THD
* ⚡ Symmetrical components
* ⚖️ Current unbalance
* 🌊 `db4` wavelet decomposition
* ⚡ Wavelet energy
* 🧠 Wavelet entropy
* 📉 Wavelet variance
* 📐 Wavelet standard deviation

---

# 🔬 MATLAB vs Python Validation

The Python inference feature extractor was validated against the original MATLAB inference feature-extraction function.

The comparison produced:

```text
Python features: 49
MATLAB features: 49

Maximum absolute difference:
1.398773193359e-01

Mean absolute difference:
8.499183338594e-03

Maximum relative difference:
3.050903350679e-11
```

The largest absolute difference occurs in a very large-magnitude wavelet-energy value.

Because the feature magnitude is very large, the relative difference is a more meaningful comparison.

The maximum relative difference is approximately:

```text
3.05 × 10⁻¹¹
```

indicating that the Python inference implementation reproduces the MATLAB feature extraction with very high numerical agreement.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive **Streamlit** dashboard for model inference.

The dashboard allows users to provide transformer current data and obtain a prediction without running the training workflow.

---

## 🎛️ Input Modes

The dashboard supports two input methods:

```text
📝 Manual Input
📄 CSV Upload
```

---

## 📝 Manual Input

The manual-input mode provides three separate input areas:

```text
Ia
Ib
Ic
```

and a separate input field for:

```text
Load Power
```

Each current signal must contain exactly:

```text
80 samples
```

The application validates the entered data before performing inference.

---

## 📄 CSV Upload

The dashboard also supports CSV-based inference.

### Required Columns

```text
Ia
Ib
Ic
Load_Power
```

An optional:

```text
Time
```

column can also be provided.

The uploaded CSV must contain exactly:

```text
80 rows
```

and `Load_Power` must remain constant throughout the uploaded window.

---

# 📥 Sample CSV

A sample inference file is included:

```text
Transformer_Data_Sample.csv
```

The sample contains:

* ⚡ 50 Hz three-phase signals
* 📡 2 kHz sampling frequency
* 📊 80 samples
* 🔌 Three-phase currents
* ⚙️ Load power

The sample can be used to test the Streamlit dashboard.

---

# 📈 Signal Visualization

The dashboard displays the three input current signals separately:

```text
Ia
Ib
Ic
```

The time axis covers:

```text
0 ms → 39.5 ms
```

This allows the user to visually inspect the input waveforms used for inference.

---

# 🔮 Prediction Result

The dashboard returns a binary classification:

### 🟢 Healthy

```text
HEALTHY
```

The model did not detect a turn-to-turn fault in the provided input window.

### 🔴 Fault

```text
FAULT DETECTED
```

The model detected a possible turn-to-turn fault in the provided input window.

The dashboard also displays the model probabilities for both classes:

```text
🟢 Healthy Probability
🔴 Fault Probability
```

---

# 📋 Extracted Features

The dashboard provides a table containing the actual features passed to the XGBoost model.

These include:

```text
49 signal-derived features
+
Load_Power
=
50 model input features
```

This makes the inference process more transparent and allows the extracted values to be inspected.

---

# ☁️ Streamlit Deployment

The dashboard can be deployed using **Streamlit Community Cloud**.

An important change in the deployment architecture is that **MATLAB is not required on the deployment server**.

Instead, the deployed application uses the Python implementation of the feature-extraction pipeline.

### 🌐 Live Demo

The deployed Streamlit application is available at:

**https://transformer-turn-to-turn-fault-detection.streamlit.app/**

You can open the dashboard directly in your browser and test the transformer fault detection inference pipeline.

### 🏗️ Training Environment

```text
MATLAB / Simulink
        ↓
Transformer Simulation
        ↓
Raw Current Signals
        ↓
MATLAB Feature Extraction
        ↓
49 Features + Load_Power
        ↓
XGBoost Training
        ↓
Saved Model
```

### ☁️ Deployment Environment

```text
User Input
   ↓
Ia / Ib / Ic
   ↓
Python Feature Extraction
   ↓
49 Features
   ↓
Load_Power
   ↓
50 Features
   ↓
Saved XGBoost Model
   ↓
Prediction
   ↓
Streamlit Dashboard
```

Therefore, MATLAB/Simulink is required for reproducing the simulation and training workflow, but **not for running the deployed Streamlit inference application**.

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/ermia-106/Transformer-Turn-to-Turn-Fault-Detection.git
```

Navigate to the project directory:

```bash
cd Transformer-Turn-to-Turn-Fault-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Dashboard

Run the following command:

```bash
streamlit run app.py
```

The Streamlit application will then open in the browser.

You can also access the deployed version directly:

🌐 **Live Demo:**
https://transformer-turn-to-turn-fault-detection.streamlit.app/

---

# 🧪 Training the Model

The original training workflow is available through:

```text
Transformer-Turn-to-Turn-Fault-Detection.py
```

or:

```text
Transformer-Turn-to-Turn-Fault-Detection.ipynb
```

The training pipeline:

1. 📥 Loads the extracted MATLAB feature dataset.
2. 🏷️ Separates the target label.
3. 🔀 Performs group-based train/test splitting.
4. 🤖 Trains the XGBoost classifier.
5. 📊 Evaluates the model.
6. 🔁 Performs GroupKFold cross-validation.
7. 🔍 Performs SHAP analysis.
8. 💾 Saves the trained model.

---

# ⚙️ MATLAB Requirements

MATLAB and Simulink are required if you want to:

* 🏗️ Reproduce the transformer simulations
* 🔄 Generate new training scenarios
* 📊 Recreate the raw training dataset
* 🧮 Recreate the original MATLAB feature extraction
* 🧪 Generate additional simulation data

The Streamlit inference dashboard does **not** require MATLAB.

---

# 🧰 Technologies

The project uses:

* ⚙️ **MATLAB**
* 🔌 **Simulink**
* 🐍 **Python**
* 🔢 **NumPy**
* 🐼 **Pandas**
* 📐 **SciPy**
* 🌊 **PyWavelets**
* 📊 **Scikit-learn**
* 🚀 **XGBoost**
* 🔍 **SHAP**
* 📈 **Matplotlib**
* 🎨 **Seaborn**
* 🖥️ **Streamlit**

---

# 📄 Main Files

## ⚙️ MATLAB

### `Generate_Transformer_Dataset.m`

Generates transformer simulation scenarios and raw three-phase current data.

### `Extract_Transformer_Features.m`

Extracts the original 49 training features from the simulated signals.

---

## 🤖 Python Training

### `Transformer-Turn-to-Turn-Fault-Detection.py`

Trains and evaluates the XGBoost model and performs SHAP analysis.

### `Transformer-Turn-to-Turn-Fault-Detection.ipynb`

Jupyter Notebook version of the machine-learning workflow.

---

## 🚀 Python Inference

### `Feature_Extraction_Inference.py`

Extracts the same 49 signal features used during training.

### `Inference.py`

Loads the saved XGBoost model, creates the 50-feature input, and performs inference.

---

## 🖥️ Streamlit

### `app.py`

Provides the interactive inference dashboard with:

* 📝 Manual input
* 📄 CSV upload
* 📈 Current waveform visualization
* 🔮 Fault prediction
* 📊 Prediction probabilities
* 📋 Extracted-feature table

---

# 🔄 End-to-End Project Workflow

```text
                    ⚙️ MATLAB / Simulink
                            │
                            ▼
                 🏗️ Transformer Simulation
                            │
                            ▼
                  🔌 Current Signals
                     Ia / Ib / Ic
                            │
                            ▼
               🧮 MATLAB Feature Extraction
                            │
                            ▼
                     📊 49 Features
                            │
                            ├──── ⚙️ Load_Power
                            │
                            ▼
                     📦 50 Features
                            │
                            ▼
                     🤖 XGBoost Model
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
              🔍 SHAP              💾 Model
                                      │
                                      ▼
                            🐍 Python Inference
                                      │
                                      ▼
                             🖥️ Streamlit App
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                       🟢 Healthy         🔴 Fault
```

---

# 🔐 Important Design Considerations

### 1. No Data Leakage

`Scenario_ID` is used for grouping but is not provided to the model.

### 2. Consistent Feature Order

The inference pipeline follows the same 49-feature order used during training.

### 3. Consistent XGBoost Version

The deployment environment uses the same XGBoost version as the local inference environment.

```text
xgboost==3.4.1
```

### 4. MATLAB-Free Deployment

The deployed Streamlit application performs feature extraction directly in Python and does not require a MATLAB installation.

### 5. Fixed Input Size

Inference operates on a single 80-sample window for each phase current.

---

# 📌 Project Status

The project currently includes:

* ✅ MATLAB/Simulink transformer simulation
* ✅ Healthy and faulty scenario generation
* ✅ Fixed-step signal resampling
* ✅ MATLAB feature extraction
* ✅ 49 signal-derived features
* ✅ `Load_Power` as the 50th model input
* ✅ XGBoost classification
* ✅ Group-based train/test splitting
* ✅ 10-Fold GroupKFold cross-validation
* ✅ SHAP explainability
* ✅ Saved XGBoost model
* ✅ Python inference feature extraction
* ✅ MATLAB/Python feature validation
* ✅ Manual Streamlit inference
* ✅ CSV-based Streamlit inference
* ✅ Three-phase waveform visualization
* ✅ Prediction probabilities
* ✅ Extracted-feature visualization
* ✅ Streamlit Community Cloud deployment

---

# 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for more information.

---

# 👨‍💻 Author

**ermia-106**

### 🔗 Links

* 💻 GitHub: https://github.com/ermia-106
* 🔗 LinkedIn: https://www.linkedin.com/in/ermia106
* 📧 Email: [ermia.sh.106@gmail.com](mailto:ermia.sh.106@gmail.com)
* 🌐 Live Demo: https://transformer-turn-to-turn-fault-detection.streamlit.app/