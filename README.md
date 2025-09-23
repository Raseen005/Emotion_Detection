# Emotion Detection

A deep learning-based Emotion Detection project that classifies facial expressions from images. This project uses Convolutional Neural Networks (CNNs) and is designed for real-time emotion recognition from images or video.

---

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Detects multiple emotions from facial images.
- Supports real-time emotion recognition.
- Trained with data augmentation for better generalization.
- Can be extended for video streams or webcam input.

---
## Model Accuracy = 72 %

## Model Architecture

The Emotion Detection model is a Residual CNN designed for 96×96 RGB facial images. It includes residual blocks, batch normalization, and dropout for better training stability and generalization.

Summary:

Input: (96, 96, 3) RGB images

Conv & Residual Blocks:

3 × Conv2D(32) + BatchNorm + ReLU + Add

MaxPooling + Dropout

3 × Conv2D(64) + BatchNorm + ReLU + Add

MaxPooling + Dropout

3 × Conv2D(128) + BatchNorm + ReLU + Add

MaxPooling + Dropout

3 × Conv2D(256) + BatchNorm + ReLU + Add

MaxPooling + Dropout

Global Average Pooling

Dense Layers:

Dense(64) + BatchNorm + Dropout

Dense(8) → Output (8 emotion classes)

Total parameters: 1,979,080

Trainable: 1,976,072

Non-trainable: 3,008

## Dataset

> ⚠️ Note: The AffectNet dataset is **not included** in this repository.  

- AffectNet is a large-scale facial expression dataset containing millions of images with emotion labels.  
- To use this project, you need to register and download AffectNet from the official website: [AffectNet Dataset](http://mohammadmahoor.com/affectnet/).  
- Preprocessing includes resizing, normalization, and optional data augmentation (rotation, flipping, zooming, shifting).  

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Raseen005/Emotion_Detection.git
cd Emotion_Detection
