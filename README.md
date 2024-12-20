# ECG Arrhythmia Detection Using Deep Learning

## Project Overview

This project uses electrocardiogram (ECG) signals to classify heartbeats into five AAMI-style categories using a one-dimensional Convolutional Neural Network (1D CNN) and RR interval features.

The goal is to explore how deep learning and ECG signal processing can help distinguish different heartbeat patterns, including normal beats and several types of arrhythmias.

**This is an educational machine learning project, not a clinically validated diagnostic tool.**

## Dataset

- **Dataset:** MIT-BIH Arrhythmia Database
- **Source:** [PhysioNet](https://physionet.org/content/mitdb/)
- **Input:** ECG heartbeat segments and RR interval features
- **Heartbeat segment length:** 252 samples
- **Sampling frequency:** 360 Hz

The data was divided into training, validation, and test sets using separate record lists.

| Dataset | Heartbeats |
|---|---:|
| Training | 40,886 |
| Validation | 10,116 |
| Testing | 47,557 |

## Heartbeat Classification

The model predicts five classes:

| Class | Description |
|---|---|
| N | Normal and selected normal-conduction beats |
| S | Supraventricular ectopic beats |
| V | Ventricular ectopic beats |
| F | Fusion beats |
| Q | Unknown or other supported beat types |

## Methodology

1. **Data acquisition:** Loaded ECG signals and annotations using WFDB.
2. **Signal processing:** Applied a bandpass filter to reduce noise.
3. **Heartbeat extraction:** Extracted 252-sample segments around annotated beats.
4. **Feature engineering:** Calculated RR intervals before and after each beat.
5. **Preprocessing:** Normalized ECG waveforms and scaled RR features using training-set statistics.
6. **Model development:** Built a 1D CNN to learn waveform patterns and combine them with RR features.
7. **Baseline comparison:** Trained a Random Forest classifier using the same training and validation split.
8. **Evaluation:** Used macro F1-score, precision, recall, accuracy, classification reports, and a confusion matrix.

## Model Architecture

The CNN includes:

- Three 1D convolutional layers
- ReLU activation functions
- Max pooling
- Adaptive average pooling
- RR feature concatenation
- Fully connected classification layers
- Dropout for regularization

The final layer produces predictions for the five heartbeat classes.

## Results

### Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 72% |
| Macro F1-score | 0.3282 |
| Weighted F1-score | 0.78 |

### Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| N | 0.96 | 0.73 | 0.83 |
| S | 0.13 | 0.12 | 0.13 |
| V | 0.44 | 0.92 | 0.60 |
| F | 0.05 | 0.82 | 0.09 |
| Q | 0.00 | 0.00 | 0.00 |

The model performed best on normal beats and achieved high recall for ventricular beats. However, it struggled to distinguish minority classes reliably.

The difference between accuracy and macro F1 highlights the impact of class imbalance. Macro F1 gives each class equal weight, making it more informative when rare heartbeat categories matter.

## Challenges and Lessons Learned

- Handling severe class imbalance across heartbeat categories.
- Aligning ECG segments, labels, and RR features correctly.
- Preventing preprocessing information from leaking from validation or test data into training.
- Understanding why accuracy alone can be misleading.
- Evaluating rare-class precision and recall instead of relying on a single metric.
- Working with PyTorch on a CPU-only environment.

## Limitations and Future Improvements

- The Q class has very few training examples, limiting meaningful evaluation.
- The F class has low precision despite high recall.
- The model has not been clinically validated.
- Further work should verify subject-level separation, refine annotation handling, improve minority-class performance, and compare models using consistent evaluation protocols.
- Additional experiments could investigate class-aware training, improved RR feature design, and alternative architectures.

## Repository Contents

- `ecg-arrhythmia-detection.ipynb` — Complete workflow for ECG preprocessing, feature engineering, model training, and evaluation.
- `models/best_ecg_cnn.pt` — Saved CNN model weights.
- `results/cnn_training_history` — Training and validation metrics across epochs.
- `results/class_distribution_summary` — Heartbeat class counts across datasets.
- `results/baseline_results` — Baseline model evaluation results.

## How to Reproduce
1. Open the notebook in Google Colab.
2. Install the required Python libraries, including WFDB, SciPy, scikit-learn, PyTorch, NumPy, Pandas, and Matplotlib.
3. Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/).
4. Run the notebook cells in order to preprocess the ECG data, train the model, and evaluate its performance.
\## Technologies Used

Python · NumPy · Pandas · SciPy · WFDB · scikit-learn · PyTorch · Matplotlib

## Disclaimer

This project is for educational and research purposes only. Its predictions must not be used to diagnose, treat, or make clinical decisions about heart conditions.

## Acknowledgments

The ECG data used in this project comes from the MIT-BIH Arrhythmia Database, available through PhysioNet.
