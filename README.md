# 👁️ Pedestrian Attribute Recognition & Tracking

**Computer Vision | Deep Learning | Python**

A computer vision system designed to detect, track, and analyze pedestrians in video sequences using Deep Learning and object tracking techniques.

Developed as a university group project during the Master's Degree in Computer Engineering, specializing in **Artificial Intelligence and Intelligent Robotics** at the University of Salerno.

## 🚀 Key Features

- **Pedestrian Detection:** Detect pedestrians in video frames using YOLOv8m.
- **Multi-Object Tracking:** Track individuals over time using BoT-SORT and Re-identification.
- **Behavior Analysis:** Monitor pedestrian movements and count directional line crossings.
- **Attribute Recognition:** Classify pedestrian attributes, including gender, bag presence, and hat presence.
- **Multitask Learning:** Use a ConvNeXt-based neural network with CBAM attention and GradNorm for attribute classification.

## 🛠️ Technologies Used

- **Language:** Python
- **Object Detection:** YOLOv8m
- **Object Tracking:** BoT-SORT, Re-identification
- **Deep Learning:** ConvNeXt, CBAM
- **Training:** Transfer Learning, Multitask Learning, GradNorm

## 📊 Results

The attribute recognition model was trained on over **93,000 images**, achieving the following classification results:

| Attribute | Accuracy |
|---|---|
| Gender Recognition | 94% |
| Bag Detection | 90% |
| Hat Detection | 98% |

These results demonstrate the effectiveness of the multitask architecture for pedestrian attribute recognition.

## ⚙️ Installation

Install the required dependencies from the repository root:

`pip install -r src/requirements.txt`

The main application is located in `src/main.py`. Video inputs, configuration files, and the required model assets must be provided to run the complete pipeline.

## 👥 Project Information

- **University:** University of Salerno
- **Course:** Artificial Vision
- **Academic Year:** 2024–2025
- **Project Type:** University Group Project
- **Team:** Andrea Rescigno, Pietro Zappullo, Mattia Lamberti, Pasquale Petti
