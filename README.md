# Enhancing-Handwritten-Text-Recognition-Using-Natural-Language-Processing-Techniques
My MSc Data Science Dissertation
# Enhancing Handwritten Text Recognition Using Natural Language Processing Techniques  
**A Novel Hybrid OCR–CRNN–NLP Approach**

**Author:** Alekya Pulikota  
**Degree:** M.Sc. Data Science  
**Institution:** University of Chester
**Supervisor:** Prof. Paul Underhill  
**Year:** 2025  

---

## Abstract

Handwritten Text Recognition (HTR) remains a challenging problem due to variability in handwriting styles, noise in scanned documents, and ambiguous character boundaries. This dissertation presents a structured, reproducible study of a hybrid deep learning and NLP-based approach for handwritten text recognition.

The research adopts a **two-stage experimental design**. In the first stage, the MNIST dataset is used as a proof-of-concept to validate preprocessing pipelines, CNN-based feature extraction, training routines, and evaluation metrics. In the second stage, the system is extended to the **IAM Handwriting Database**, where a **Convolutional Recurrent Neural Network (CRNN)** combined with **Connectionist Temporal Classification (CTC)** decoding is applied to real-world handwritten text.

A novel contribution of this work is the integration of **Natural Language Processing (NLP)** techniques—such as tokenization, lemmatization, and Named Entity Recognition (NER)—as a post-processing layer to refine OCR outputs. Model performance is evaluated using **Character Error Rate (CER)** and **Word Error Rate (WER)**, demonstrating that contextual NLP can reduce semantic and structural errors in handwritten text recognition.

---

## Keywords

Handwritten Text Recognition, OCR, CRNN, CNN, BiLSTM, CTC, NLP, MNIST, IAM Dataset, CER, WER

---

## Repository Structure

```text
.
├── data/
│   ├── mnist/                 # MNIST dataset (loaded via Keras)
│   ├── iam/                   # IAM handwriting images and annotations (external)
│   ├── sample_records.csv     # Example structured outputs
│   └── sample_records.json
├── src/
│   ├── mnist_cnn.py            # CNN model for MNIST
│   ├── crnn_iam.py             # CRNN + CTC for IAM dataset
│   ├── nlp_postprocessing.py   # NLP correction and NER
│   └── evaluation.py           # CER / WER computation
├── notebooks/
│   └── experiments.ipynb       # Exploratory and visualization notebook
├── results/
│   ├── figures/               # Accuracy, loss, CER/WER plots
│   └── tables/                # Quantitative evaluation results
├── README.md
└── requirements.txt
**Introduction**

Despite advances in OCR, handwritten documents remain difficult to digitize accurately due to inter- and intra-writer variability, cursive writing, and document degradation. This project investigates whether combining deep learning–based HTR models with language-level NLP correction can improve recognition accuracy and usability in real-world scenarios such as healthcare, legal documentation, and archival digitization.
Methodology Overview
Two-Stage Experimental Design

Stage 1: MNIST (Proof of Concept)

Validates CNN architecture

Confirms preprocessing, training, and evaluation pipeline

Uses digit classification accuracy as baseline

Stage 2: IAM Handwriting Database (Core Experiment)

Applies CRNN with CNN + BiLSTM layers

Uses CTC loss for alignment-free transcription

Introduces NLP post-processing for semantic correction

Datasets
MNIST Dataset

70,000 grayscale digit images (28×28)

60,000 training / 10,000 test samples

Purpose: controlled validation of model pipeline

Loaded automatically via TensorFlow/Keras:
from tensorflow.keras.datasets import mnist
IAM Handwriting Database

1,539 scanned pages of handwritten English text

600+ writers

Annotated at character, word, and line level

Used for real-world handwritten text recognition

Note: IAM dataset must be requested separately from the official source
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Data Preprocessing

Image resizing (fixed height, preserved aspect ratio)

Grayscale conversion

Normalization to [0,1]

Noise reduction (Gaussian blur, thresholding)

Data augmentation:

Rotation

Scaling

Elastic distortion

Translation

These steps ensure robustness to handwriting variability and scanning artifacts.

Model Architecture
CRNN Pipeline
Input Image
   ↓
CNN (Spatial Feature Extraction)
   ↓
BiLSTM (Sequential Modeling)
   ↓
CTC Decoder (Alignment-Free Transcription)
   ↓
NLP Post-Processing (Error Correction + NER)
Key Components

CNN: Extracts stroke-level visual features

BiLSTM: Models left-to-right and right-to-left dependencies

CTC Loss: Handles variable-length sequences without segmentation

NLP Layer: Improves semantic consistency and entity recognition

NLP Post-Processing

NLP techniques are applied after OCR output:

Tokenization and lemmatization

Context-based word correction

Named Entity Recognition (NER)

Domain-aware refinement (e.g., names, dates, identifiers)

Example:
OCR Output:  "paracetarnol"
NLP Output:  "paracetamol"
Evaluation Metrics

Accuracy (MNIST)

Character Error Rate (CER)

Word Error Rate (WER)

Entity-level correctness (post-NLP)

Evaluation scripts are located in:
src/evaluation.py


*** How to Run****
Environment Setup
pip install -r requirements.txt

Train MNIST CNN
python src/mnist_cnn.py

Train CRNN on IAM Dataset
python src/crnn_iam.py

Apply NLP Post-Processing
python src/nlp_postprocessing.py

Compute CER / WER
python src/evaluation.py

Results

MNIST CNN achieves ~99% test accuracy

CRNN significantly reduces CER and WER on IAM dataset

NLP post-processing improves semantic accuracy and reduces domain-critical errors

Results, plots, and tables are stored in:

results/

Limitations

Computationally intensive training on IAM dataset

English-only dataset scope

NLP correction effectiveness depends on domain vocabulary

IAM dataset access restrictions

Ethical Considerations

No sensitive personal data used

IAM dataset used under academic research license

GDPR considerations discussed in dissertation

Reproducibility Statement

All experiments can be reproduced using:

Provided scripts

Fixed preprocessing steps

Standard datasets (MNIST, IAM)

Reported hyperparameters

Random seeds and environment details are documented in the dissertation.

Citation

If you use this work, please cite:

Pulikota, A. (2025). Enhancing Handwritten Text Recognition Using Natural Language Processing Techniques. University of Chester.

License

Code: MIT License

Datasets: Subject to original dataset licenses (MNIST, IAM)

Acknowledgements

This work was completed under the supervision of Prof. Paul Underhill.
The author acknowledges the University of Chester and supporting faculty.


