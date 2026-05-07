# Real-Time EEG Motor Imagery Decoding Simulation

This repository demonstrates a **real-time brain–computer interface (BCI) decoding pipeline** using EEG motor imagery data. It extends previous work from offline classification toward **latency-aware, streaming inference** using sliding windows.

## Project Evolution

This work is part of a structured progression:

1- [Baseline CSP pipeline](https://github.com/SanazRezvani/eeg-motor-imagery-csp)
 
2- [FBCSP extension](https://github.com/SanazRezvani/eeg-motor-imagery-fbcsp)  

### 3- **Current project:** Real-time decoding simulation

This work is based on **BCI Competition III – Dataset IVa**. Read the [Dataset description](https://www.bbci.de/competition/iii/desc_IVa.html)

---
## Objective

Most EEG classification pipelines are evaluated offline using labelled trials. However, real-world BCI systems must:

- Process continuous EEG streams
- Generate predictions in real time
- Operate under latency constraints

This project simulates that transition by:

- Applying sliding window segmentation
- Performing real-time feature extraction and classification
- Evaluating Latency and System Feasibility
- Analysing temporal prediction behaviour

## Two-Phase Pipeline
### 1- Offline phase:
- Apply Filter Bank Common Spatial Pattern (FBCSP) and extract features across multiple sub-bands using labelled trials

### 2- Real-time simulation phase:
- Treat test trials as an incoming EEG stream
- Apply sliding windows
- Extract FBCSP features per window
- Generate predictions
- Log processing latency

## How to Run

- Download the dataset from [BCI Competition III – Dataset IVa](https://www.bbci.de/competition/iii/)

- Open MATLAB

- Start by loading one of the subjects. ` data_set_IVa_al.mat ` is chosen here.

- Run: ` run_realtime_simulation.m `

Inside `run_realtime_simulation.m`, you can modify:
``` 
config.dataset_path = 'data_set_IVa_al.mat';
config.spatial_filter = 'CAR';       % options: 'CAR', 'Low Laplacian', 'High Laplacian'
config.filter_order = 3;
config.train_ratio = 0.70;
config.num_csp_pairs = 1;
config.trial_length_s = 3.5;
config.window_length_s = 1.0;
config.step_size_s = 0.25;
config.classifier_type = 'KNN';      % options: 'LDA', 'SVM', 'KNN'
config.filter_bank = [8 12; 10 14; 12 16; 14 18; 16 20; 18 22; 20 24; 22 26; 24 28; 26 30];
```

## Notes on the offline phase

Motor imagery EEG activity is distributed across multiple frequency ranges, primarily within the mu (8–13 Hz) and beta (13–30 Hz) bands and the most informative frequency sub-bands can vary between subjects. Applying CSP on a single broad band may overlook frequency-specific patterns 

Filter Bank CSP (FBCSP) addresses this by:
- Decomposing EEG signals into multiple sub-bands
- Extracting spatial features from each sub-band
- Capturing complementary information across frequencies

This leads to a richer feature representation and can improve classification performance, especially in subject-specific decoding scenarios.

### Key Features

- Extension of a baseline CSP-based motor imagery decoding pipeline
- Filter bank decomposition across mu and beta rhythms (8–30 Hz)
- 10 overlapping sub-bands (4 Hz width, 2 Hz overlap)
- CSP feature extraction from each sub-band
- Concatenation of sub-band features
- Classification of motor imagery tasks using machine learning

### Design Choices

- **Sub-band design:** 4 Hz bandwidth with 2 Hz overlap to balance frequency resolution and redundancy  
- **Number of CSP pairs:** 1 pair per sub-band (2 features) to control dimensionality  
- **Classifier selection:** Compared SVM, KNN, and LDA for robustness  
- **Feature representation:** Variance of CSP-projected signals, a standard and interpretable choice for motor imagery BCI  

These design decisions aim to balance performance, interpretability, and computational efficiency.

### Extracting CSP features from each sub-band

Implemented in: 
[`compute_csp_filters.m`](compute_csp_filters.m) (compute_csp_filters) 

For each frequency sub-band, Common Spatial Pattern (CSP) is applied to extract discriminative spatial features between the two motor imagery classes.

CSP computes spatial filters that maximise variance for one class while minimising it for the other. This results in projections that emphasise class-specific neural activity.
```
features_band(:, trial_idx) = var(projected_trial, 0, 2);
```

In this implementation:

CSP is computed independently for each sub-band
A fixed number of CSP filter pairs are selected per band
Each trial is projected onto the CSP filters
The variance of the projected signals is used as the feature representation

This allows the model to capture frequency-specific spatial patterns, which are critical in motor imagery EEG analysis.

![CSP Animation](results/csp_animation.gif)

### Concatenating features across sub-bands

The CSP features extracted from each sub-band are concatenated to form a single feature vector for each trial.

Since each sub-band captures complementary information from different parts of the mu and beta frequency ranges, combining them provides a richer representation of the underlying neural activity.

In this project:

Each sub-band contributes a set of CSP features
Features from all sub-bands are stacked vertically
The final feature vector includes information from all frequency bands
```
all_features = [features_band1;
                features_band2;
                ...
                features_bandN];
```
In this implementation:
```
10 sub-bands × 2 CSP features = 20 features per trial
```

### Training and evaluating classifiers

One of the following classifiers can be chosen:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)

## Notes on the online phase

### Sliding Window Design

- Window length	= 1.0 s
- Step size =	0.25 s
- Overlap	= 75%

This allows the system to generate predictions every 250 ms while using sufficient temporal context for feature extraction.

## Results
Observations
- Initial latency spike (~10 ms) due to system warm-up
- Stabilises around 2.6–3 ms
- The **Streaming accuracy** is **82.93 %**

The system demonstrates strong real-time capability, as processing latency is significantly lower than the step size.

![Latency over time](results/latency_over_time.png)


### Prediction Log
| WindowIndex | Time (s) | Predicted | True | Latency (ms) |
|-------------|----------|----------|------|--------------|
| 1 | 0.01 | 2 | 1 | 10.594875 |
| 2 | 0.26 | 1 | 1 | 7.09304166666667 |
| 3 | 0.51 | 1 | 1 | 5.99429166666667 |
| 4 | 0.76 | 1 | 1 | 6.52583333333333 |
| 5 | 1.01 | 1 | 1 | 7.26666666666667 |
| 6 | 1.26 | 1 | 1 | 5.82733333333333 |
| 7 | 1.51  | 1 | 1 | 4.95966666666667 |
| 8 | 1.76 | 1 | 1 | 6.1315 |
| 9 | 2.01 | 2 | 1 | 5.76120833333333 |
| 10 | 2.26 | 2 | 1 | 4.51675 |
| 11 | 2.51 | 2 | 1 | 4.0345 |
| 12 | 2.76 | 1 | 1 | 4.07366666666667 |
| 13 | 3.01 | 1 | 1 | 4.02595833333333 |
| 14 | 3.26 | 1 | 2 | 3.43941666666667 |
| 15 | 3.51  | 2 | 2 | 3.23741666666667 |
| 16 | 3.76 | 2 | 2 | 2.72808333333333 |
| 17 | 4.01 | 2 | 2 | 3.19816666666667 |
| 18 | 4.26 | 2 | 2 | 3.03316666666667 |
| 19 | 4.51  | 2 | 2 | 3.29754166666667 |
| 20 | 4.76 | 2 | 2 | 3.10691666666667 |
| 21 | 5.01 | 2 | 2 | 2.770625 |

Full log available here: [prediction_log.csv](results/prediction_log.csv)

## Key takeaway:  
> While the system achieves low latency and continuous decoding, prediction stability remains a critical challenge, highlighting the importance of temporal smoothing and system-level design in real-world BCI applications.
