# Real-Time EEG Motor Imagery Decoding Simulation

This repository demonstrates a **real-time brain–computer interface (BCI) decoding pipeline** using EEG motor imagery data. It extends previous work from offline classification toward **latency-aware, streaming inference** using sliding windows.


## 🔗 Project Evolution

This work is part of a structured progression:

1- [Baseline CSP pipeline](https://github.com/SanazRezvani/eeg-motor-imagery-csp)
 
2- [FBCSP extension](https://github.com/SanazRezvani/eeg-motor-imagery-fbcsp)  

### 3- **Current project:** Real-time decoding simulation

🎯 Objective

Most EEG classification pipelines are evaluated offline using pre-segmented trials. However, real-world BCI systems must:

Process continuous EEG streams
Generate predictions in real time
Operate under latency constraints

This project simulates that transition by:

Applying sliding window segmentation
Performing real-time feature extraction and classification
Measuring processing latency
Analysing temporal prediction behaviour
⚙️ System Overview
🔁 Two-Phase Pipeline
1. Offline Phase (Training)
Apply Filter Bank CSP (FBCSP)
Extract features across multiple sub-bands
Train classifier (LDA / SVM / KNN)
2. Real-Time Phase (Simulation)
Simulate continuous EEG stream
Apply sliding windows
Extract FBCSP features per window
Perform classification
Log predictions and latency
🧩 Sliding Window Design
Parameter	Value
Window length	1.0 s
Step size	0.25 s
Overlap	75%

This allows the system to generate predictions every 250 ms while using sufficient temporal context for feature extraction.

📊 Results
⚡ Latency Over Time

🔍 Observations
Initial latency spike (~10 ms) due to system warm-up
Stabilises around 2.6–3 ms
Occasional small fluctuations
🧠 Interpretation

The system demonstrates strong real-time capability, as processing latency is significantly lower than the step size:


