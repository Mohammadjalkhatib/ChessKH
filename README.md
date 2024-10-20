# Determining Chess Board State Using Computer Vision Techniques to Predict the Next Best Move

## Authors:
- **Mohammad Jamal AlKhatib** - Faculty of Engineering, University of Jordan (Masters in AI and Robotics)
- **Moath Rabeh Khaleel** - Faculty of Engineering, University of Jordan (Masters in AI and Robotics)

## Abstract:
This project focuses on designing an image classification system to determine the state of a chessboard. The determined state can then be fed into a chess engine like Stockfish to compute the next best move. By using pre-trained models for image classification, we aim to create an efficient system that can automate the process of feeding board states to the engine, enabling the use of computer vision for enhanced chess gameplay.

## Introduction:
Artificial Intelligence (AI) has made significant strides in many areas, including chess. Since the 20th century, AI has been used to create chess engines capable of outperforming human players. This project bridges the gap between human input and chess engines by automating board state recognition using computer vision techniques. This system has the potential to act as a chess trainer or be integrated with a robotic chess-playing system.

## Approach:
The approach involves three major steps:
1. **Enhancing the clarity of the chessboard image** using techniques like Canny edge detection and Hough Transform.
2. **Segmenting the board into 64 squares** to detect the position and type of pieces.
3. **Classifying chess pieces** by using pre-trained models like VGG16, InceptionV3, and ResNet.

### System Architecture:
- **Image Segmentation:** Board images are divided into 64 squares using edge detection.
- **Piece Classification:** Pre-trained deep learning models are used to classify the pieces occupying each square.
- **Chess Engine Integration:** The detected board state is fed to Stockfish to compute the next best move.

## Data Description:
A dataset was created by capturing 2,235 real-world images of chess pieces from various angles and under different lighting conditions. The dataset consists of 13 classes (e.g., black pawn, white king, empty square). Pre-trained models were applied to classify each image, with transfer learning used to enhance the models.

## Implementation:
Python 3.11 was used for the entire implementation process. Key libraries included:
- **OpenCV** for image processing.
- **TensorFlow** for model training and optimization.

Transfer learning was employed with models such as VGG16, InceptionV3, and ResNet. Techniques like dropout, regularization, and data augmentation were used to enhance model performance.

## Results:
Different model architectures were tested. The best performance was achieved using InceptionV3 with transfer learning, which resulted in a validation accuracy of **97.25%**. The table below summarizes the results:

| **Model**                     | **Trainable Parameters** | **Accuracy** | **Validation** |
|-------------------------------|--------------------------|--------------|----------------|
| InceptionV3 (Transfer Learning)| 21,936,429                | 99.89%       | 97.25%         |
| VGG16 (Transfer Learning)      | 4,203,917                 | 93.90%       | 85.59%         |

## Conclusion:
The proposed system shows high accuracy in determining the board state. However, further improvements can be made, especially in the board segmentation step. Future work could involve refining the segmentation algorithm and improving the overall accuracy through additional layers or computational power.

## Future Work:
- Implementing an advanced board segmentation algorithm.
- Improving the classification accuracy.
- Integrating the system with a physical robot for full automation in chess gameplay.

---

### How to Run the Project:

1. **Install Required Libraries**:
   ```
   pip install tensorflow opencv-python numpy
   ```

2. **Dataset Preparation**:
   - Place your chessboard images in the `dataset` directory.
   - Ensure images are labeled correctly according to the 13 predefined classes.

3. **Training the Model**:
   - To train the model, run:
     ```bash
     python train_model.py
     ```

4. **Testing the Model**:
   - Use the following command to test on new images:
     ```bash
     python test_model.py --image_path path_to_image
     ```

### References:
1. Kasparov, G. (1997). The Match of the Century.
2. Wolflein, A., et al. (2020). "Image Processing in Chess Board Recognition."
3. Chen, W., et al. (2021). "Computer Vision Techniques for Chess Board Recognition."
4. Stockfish Developers. (2023). "Stockfish: The UCI Chess Engine."
