Below is a revised README file that reflects your actual project:

---

# Automated Chess-Playing System Using YOLOv8, Stockfish, and Quanser QArm

**Author:**  
**Mohammad Jamal AlKhatib** – AI Engineer

---

## Abstract

This project presents a fully automated chess-playing system that integrates computer vision, artificial intelligence, and robotics. The system leverages a state-of-the-art YOLOv8x model to detect chess pieces and board configurations, converts detections into Forsyth–Edwards Notation (FEN), and uses the Stockfish chess engine to compute the optimal move. A Quanser QArm robotic manipulator then executes the move with high precision. Experimental results in a controlled lab environment demonstrate a YOLOv8x detection performance with precision, recall, and F1 scores of approximately 0.977, 0.974, and 0.975 respectively (validation), while real-time testing yielded an 85.8% detection accuracy. The QArm achieved an 87% success rate on the first attempt, with only minor adjustments required in subsequent attempts.

---

## Keywords

Human-Computer Interaction, Computer Vision, Robotics, Chessboard Detection, Stockfish

---

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Computer Vision Model](#computer-vision-model)
- [Robotics Integration](#robotics-integration)
- [Experimental Setup & Results](#experimental-setup--results)
- [How to Run the Project](#how-to-run-the-project)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Advancements in AI-driven Human-Computer Interaction (HCI) have paved the way for systems that not only understand human intent but also execute complex tasks autonomously. In this project, chess is used as a testbed to demonstrate the integration of:
- **Computer Vision:** Using YOLOv8x for real-time detection of chess pieces and board states.
- **Decision-Making:** Converting detection results into FEN notation and computing moves with Stockfish.
- **Robotic Actuation:** Precisely executing moves on a physical chessboard using the Quanser QArm.

This end-to-end system bridges digital perception with physical interaction, offering potential applications in assistive robotics, industrial automation, and beyond.

---

## System Architecture

The automated chess-playing pipeline comprises three main modules:

1. **Vision Module:**  
   - A top-view camera captures live images of the chessboard.
   - The YOLOv8x model detects both the chess pieces and the board outline.
   - An algorithm converts these detections into FEN notation, accurately representing the current board state.

2. **Decision Module:**  
   - The FEN string is fed into the Stockfish chess engine, which computes the optimal move based on the current game state.

3. **Actuation Module:**  
   - A mapping algorithm converts Stockfish’s move into physical (X, Y, Z) coordinates.
   - The Quanser QArm robotic manipulator, controlled via MATLAB/Simulink, executes the move following a predefined “home, pick, home, place, home” trajectory.

A simplified flow of the system is illustrated below:

```
[Camera] --> [YOLOv8x Detection] --> [FEN Conversion] --> [Stockfish]
      \                                               /
       \________________[Mapping Algorithm]_________/
                          |
                      [QArm Execution]
```

---
## Dataset

A custom dataset was created to meet the specific needs of chess piece and board detection with YOLOv8:

- **Dataset Composition:**  
  - **Part 1:** 250 images captured from five different angles using a Xiaomi 12T Pro smartphone.
  - **Part 2:** 173 top-view images from iconic chess matches (e.g., Kasparov vs. Topalov).

- **Annotation & Augmentation:**  
  - Images were annotated using [Roboflow](https://roboflow.com), with automated labeling to ensure consistency.
  - Data augmentation included horizontal/vertical flips, brightness adjustments, blur, and noise addition.
  - A unique "Chess_Board" class was introduced in the second dataset to aid in accurate board localization.

- **Class Distribution:**  
  The datasets include standard chess pieces (both colors) along with the board class, ensuring robust training for the YOLOv8x model.

This repository contains two complementary datasets: `chess_dataset_topview` and `chess_mydata`. These datasets were prepared and processed using Roboflow, focusing on chess piece detection and analysis from a top-down perspective. The `chess_dataset_topview` provides bird's-eye view images of chess configurations, while `chess_mydata` supplements this with additional chess-related data. These datasets are suitable for computer vision tasks such as chess piece detection, board position analysis, and automated chess game tracking.

### How to Use the Data

1. **Direct Download**: Clone this repository using:
```bash
git clone https://github.com/Mohammadjalkhatib/ChessKH
```

2. **Roboflow Integration**: Access the datasets programmatically using Roboflow:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="your-api-key")

# Load chess_dataset_topview
chess_topview = rf.workspace().project("chess_dataset_topview").version(1).download("yolov5")

# Load chess_mydata
chess_mydata = rf.workspace().project("chess_mydata").version(1).download("yolov5")
```

3. **Dataset Format**: The data is provided in YOLOv5 format, which includes:
   - Images in `.jpg` or `.png` format
   - Labels in `.txt` files
   - `data.yaml` configuration file

---

## Computer Vision Model

### YOLOv8x Training & Configuration

- **Model:** YOLOv8x (chosen for its superior detection capabilities in complex scenarios)
- **Training Setup:**
  - **Epochs:** 75  
  - **Image Size:** 640x640  
  - **Batch Size:** 16  
  - **Optimizer:** Adam with a learning rate of 0.01  
  - **Hardware:** TPU with AMP enabled for resource efficiency
  - **Augmentations:** Blur, color adjustments, brightness variations, etc.

### Performance Metrics

- **Validation (26 images, 614 instances):**
  - **Precision:** 0.977
  - **Recall:** 0.974
  - **F1 Score:** 0.975
  - **mAP50:** 0.988
  - **mAP50-95:** 0.867

- **Test Dataset:**  
  Detailed per-class performance showed near-perfect precision and recall for critical classes like Black_Pawn and Chess_Board.

A key innovation was the conversion of YOLOv8x detection outputs into FEN notation, enabling seamless integration with Stockfish.

---

## Robotics Integration

### Quanser QArm Overview

- **Configuration:** 4-DOF robotic arm with revolute joints (base, shoulder, elbow, wrist)
- **Actuation:**
  - Uses Dynamixel servo motors and a tendon-based gripper with two articulated fingers.
  - Controlled via MATLAB/Simulink with Quanser’s QUARC software.
  - Predefined waypoints ensure reliable “home, pick, home, place, home” movements.

### Calibration & Mapping

- **Chessboard Setup:**  
  - Positioned 25 cm from the QArm.
  - Each square measures approximately 35x35 cm.
  - Reference points on the board allow translation of image coordinates into physical (X, Y, Z) coordinates.
- **Gripper Adjustment:**  
  - Initial gripper opening is decreased to secure chess pieces during pick-and-place operations.

---

## Experimental Setup & Results

### Experimental Setup

- **Environment:** Controlled lab setting with minimized lighting fluctuations.
- **Procedure:**  
  - A top-view camera provided real-time images.
  - The system was evaluated over 100 trials, tracking both detection accuracy and robotic actuation performance.

### Results

- **Computer Vision:**  
  - Controlled conditions: YOLOv8x achieved near-perfect detection.
  - Real-time testing: Average detection accuracy was 85.8%, impacted by variations in lighting and camera positioning.
  
- **Robotic Actuation:**  
  - 87% of moves were successfully executed on the first attempt.
  - 10% required a second attempt.
  - Less than 3% needed manual board adjustment.

These results demonstrate robust integration across modules and competitive performance compared to related systems.

---

## How to Run the Project

### Prerequisites

- **Hardware:**  
  - Top-view camera  
  - Chessboard  
  - Quanser QArm robotic manipulator  
  - Laptop/PC with MATLAB/Simulink for QArm control

- **Software:**  
  - Python 3.11  
  - MATLAB/Simulink (with QUARC)  
  - [TensorFlow](https://www.tensorflow.org/)  
  - [OpenCV](https://opencv.org/)  
  - [YOLOv8](https://github.com/ultralytics/ultralytics)  
  - Stockfish chess engine

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/automated-chess-system.git
   cd automated-chess-system
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install tensorflow opencv-python numpy
   ```

3. **Prepare the Dataset:**
   - Place your annotated chessboard images in the designated dataset directory.
   - Ensure that the annotations follow YOLOv8 requirements (or use the provided Roboflow dataset).

### Running the Modules

1. **Training the YOLOv8x Model:**
   ```bash
   python train_yolov8.py --data_path ./dataset --epochs 75 --img_size 640
   ```

2. **Testing the Vision Module:**
   ```bash
   python test_yolov8.py --image_path path_to_test_image
   ```

3. **Full Pipeline Execution:**
   - Run the integrated script that:
     - Captures images from the camera.
     - Performs YOLOv8x detection.
     - Converts detections to FEN notation.
     - Invokes Stockfish for move prediction.
     - Maps moves to physical coordinates.
   - This script then sends the command to the QArm via MATLAB/Simulink.
   ```bash
   python run_full_pipeline.py --camera_index 0
   ```

---

## Future Work

- **Enhanced Board Segmentation:**  
  Refine segmentation algorithms to further improve detection under varied lighting and perspective conditions.

- **Real-Time Adjustment Algorithms:**  
  Develop feedback mechanisms to correct detection errors dynamically during live play.

- **Robotic Calibration:**  
  Improve calibration methods for a permanently fixed setup, potentially increasing actuation accuracy.

- **Extended Applications:**  
  Explore the system’s application in other HCI domains such as industrial automation and assistive robotics.

---

## References

1. Bennet, A. & Lasen, B. – Chessboard corner detection techniques.  
2. Liu, et al. – Hessian-based corner detection improvements.  
3. Mallasén Quintana, et al. – LiveChess2FEN framework using CNNs.  
4. Wölflein & Arandjelović – Accurate chess piece classification in varied conditions.  
5. Masouris & van Gemert – End-to-end chess recognition with deep learning.  
6. Matuszek, et al. – Gambit: Robotic chess piece manipulation.  
7. Chen & Wang – Autonomous chess analysis using humanoid robotics.  
8. Quanser QArm specifications and documentation.

