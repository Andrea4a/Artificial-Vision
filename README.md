# Pedestrian Attribute Recognition & Tracking

**Computer Vision · Deep Learning · Multi-Object Tracking · Multitask Learning**

An end-to-end Computer Vision system for detecting, tracking and analyzing pedestrians in video sequences, combining object detection, multi-object tracking and deep learning-based attribute recognition.

Developed as an academic group project during the Master's Degree in Computer Engineering, specializing in Artificial Intelligence and Intelligent Robotics at the **University of Salerno**.

## 🎯 Project Overview

The system processes surveillance video to identify pedestrians, analyze their movements and recognize visual attributes.

**Main functionalities:**
- **Pedestrian Detection:** Identify pedestrians using YOLOv8m.
- **Multi-Object Tracking:** Maintain consistent identities across video frames using BoT-SORT and re-identification.
- **Behavior Analysis:** Detect and count pedestrian crossings along predefined directional lines.
- **Attribute Recognition:** Predict three pedestrian attributes: gender, bag presence and hat presence.
- **Structured Output:** Export video analysis results in JSON format.

## 🧠 Architecture & Technical Approach

The system combines detection, tracking and multitask classification in a unified video processing pipeline.

**1. Detection & Tracking**

YOLOv8m generates pedestrian bounding boxes, while BoT-SORT performs multi-object tracking with re-identification to maintain individual identities across frames.

**2. Pedestrian Behavior Analysis**

Tracked trajectories are analyzed to detect line crossings and evaluate their direction, enabling directional pedestrian counting.

**3. Multitask Attribute Recognition**

A deep learning model based on **ConvNeXt Base** performs feature extraction through a shared backbone.

The architecture integrates:
- **CBAM Attention:** Enhances relevant visual features through channel and spatial attention.
- **Task-Specific Classification Heads:** Three separate branches for pedestrian attribute prediction.
- **Transfer Learning:** Fine-tuning of pre-trained features for pedestrian analysis.
- **GradNorm:** Dynamic loss weighting to balance the different classification tasks.
- **Masked Losses:** Handling missing dataset annotations during training.

To improve prediction reliability, attribute recognition is performed periodically and aggregated over time. Predictions on heavily overlapping pedestrians are deferred to reduce ambiguous classifications.

## 📊 Experimental Results

The multitask model was trained and evaluated using an annotated pedestrian attribute dataset.

| Metric | Reported Result |
|---|---|
| Training Images | 93,081 |
| Validation Images | 12,161 |
| Attribute Recognition F1-score | Approximately 90% |
| Gender Test Loss | 0.1935 |
| Bag Test Loss | 0.3607 |
| Hat Test Loss | 0.0745 |

The reported F1-score summarizes the attribute recognition performance, not the accuracy of pedestrian detection or tracking.

**Challenges and limitations**

- Identity switches during heavy occlusions.
- Reduced tracking reliability for distant pedestrians.
- Class imbalance and missing annotations in the attribute dataset.

## 🛠️ Technologies

**Programming:** Python

**Computer Vision:** YOLOv8m, BoT-SORT, Object Detection, Multi-Object Tracking, Re-identification

**Deep Learning:** ConvNeXt, CBAM, Transfer Learning, Multitask Learning, GradNorm

## 📁 Repository Structure

- `src/main.py` — Application entry point
- `src/inference.py` — Inference module
- `src/feature_extractor.py` — Feature extraction
- `src/classification_head.py` — Attribute classification
- `src/multitask_nn.py` — Multitask neural network
- `src/mapping.py` — Mapping module
- `src/utils/` — Supporting utilities
- `src/requirements.txt` — Project dependencies

## ⚙️ Installation & Usage

Install the required Python dependencies:

`pip install -r src/requirements.txt`

Example execution:

`python src/main.py --video video/video.mp4 --results results/results.txt --configuration configuration/config.txt`

The paths are illustrative and must be adapted to the input files. Required configuration files and model assets must be available. Execution instructions should be verified against the current code.

## 🔮 Future Improvements

- Improve re-identification under occlusions.
- Optimize the pipeline for real-time inference.
- Extend recognition to additional pedestrian attributes.
- Evaluate robustness across different surveillance environments.

## 👨‍💻 Project Information

**Academic Project:** Pedestrian Attribute Recognition

**Institution:** University of Salerno

**Program:** Master's Degree in Computer Engineering — Artificial Intelligence and Intelligent Robotics

**Development:** Collaborative university project involving Andrea Rescigno and team members.
