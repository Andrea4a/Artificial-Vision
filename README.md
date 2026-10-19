# Pedestrian Attribute Recognition & Tracking

**Computer Vision | Deep Learning | Multi-Object Tracking | Multitask Learning**

An end-to-end computer vision system designed to detect, track and analyze pedestrians in surveillance videos, combining object detection, multi-object tracking, behavior analysis and deep learning-based attribute recognition.

Developed as a team project during the Master's Degree in Computer Engineering at the **University of Salerno**.

## 🎯 Project Overview

The system integrates multiple computer vision components to:

- Detect pedestrians using **YOLOv8m**.
- Track individuals across video frames with **BoT-SORT and Re-identification**.
- Analyze trajectories and count directional line crossings.
- Recognize pedestrian attributes: gender, bag presence and hat presence.
- Generate structured video analysis results.

<img width="526" height="291" alt="image" src="https://github.com/user-attachments/assets/3da14e6f-2d5f-4155-9880-0fdef86149e5" />

## 🧠 Technical Architecture

### 1. Pedestrian Detection & Tracking

**YOLOv8m** is used for pedestrian detection, while **BoT-SORT** maintains consistent identities across frames.

An OSNet-based Re-identification model supports identity association after temporary occlusions.

### 2. Behavior Analysis & Line Mapping

Pedestrian movements are analyzed to identify crossings of two predefined directional lines.

The system also implements a 3D-to-2D projection approach based on the pinhole camera model, rotation matrices and camera parameters to map directional lines onto the image plane.

### 3. Multitask Deep Learning

The attribute recognition model uses a **ConvNeXt Base** backbone with a shared feature extraction architecture and three task-specific classification heads.

Main architectural components:

- **ConvNeXt Base:** Shared feature extraction with transfer learning.
- **CBAM Attention:** Channel and spatial attention for improved feature representation.
- **Multitask Learning:** Simultaneous prediction of three pedestrian attributes.
- **GradNorm:** Adaptive task-loss balancing during training.
- **Masked Loss:** Handling missing attribute annotations.

### 4. Temporal Attribute Aggregation

Attribute predictions are periodically collected during pedestrian tracking and aggregated using majority voting.

The system also uses a confidence strategy based on bounding-box overlap, postponing classification when the Intersection over Union (IoU) exceeds 0.30.

## 📊 Experimental Results

The multitask architecture was trained using:

- **93,081 training images**
- **12,161 validation images**
- **Optimizer:** AdamW
- **Batch size:** 128
- **Data augmentation:** Horizontal flipping and random brightness/contrast adjustments

### Classification Performance

| Attribute | Accuracy | Weighted F1-score |
|---|---|---|
| Gender | 94% | 0.94 |
| Bag Presence | 90% | 0.90 |
| Hat Presence | 98% | 0.98 |

The model achieved strong classification results, although class imbalance affected performance on underrepresented categories.

In particular, the Bag class obtained an F1-score of 0.70, illustrating the importance of evaluating individual class performance alongside aggregated metrics.

## 🛠️ Technologies

**Programming:** Python

**Computer Vision:** YOLOv8, BoT-SORT, Object Tracking, Re-identification

**Deep Learning:** ConvNeXt, CBAM, Transfer Learning, Multitask Learning

**Training Techniques:** AdamW, GradNorm, Data Augmentation, Masked Losses

## 📁 Project Structure

- `src/main.py` — Main application
- `src/inference.py` — Inference pipeline
- `src/feature_extractor.py` — Feature extraction
- `src/classification_head.py` — Attribute classification heads
- `src/multitask_nn.py` — Multitask neural network
- `src/mapping.py` — Mapping utilities
- `src/utils/` — Supporting modules
- `src/requirements.txt` — Dependencies

## ⚙️ Installation

Install the project dependencies:

`pip install -r src/requirements.txt`

Additional input videos, configuration files and model assets may be required to run the complete pipeline.

## 🔍 Challenges & Future Improvements

- Handling identity switches in crowded scenes.
- Improving detection and tracking of distant pedestrians.
- Addressing class imbalance in attribute recognition.
- Optimizing inference for real-time applications.

## 👥 Project Team

**University of Salerno — Master's Degree in Computer Engineering**

Academic Year 2024–2025

- Andrea Rescigno
- Pietro Zappullo
- Mattia Lamberti
- Pasquale Petti

Collaborative academic project developed for the Artificial Vision course.
