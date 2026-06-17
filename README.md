#  MNIST Digit Classifier (CNN)

An end-to-end Handwritten Digit Recognition system built using PyTorch. This project uses a Convolutional Neural Network (CNN) to recognize digits (0-9) with high accuracy.

##  Features
- **Deep Learning Model:** Custom CNN architecture with Batch Normalization and Dropout to prevent overfitting.
- **Smart Preprocessing:** 
  - **Automatic Bounding Box Cropping:** Digit ko canvas se crop karke center karta hai taaki stretching na ho.
  - **Gaussian Blurring:** Sharp lines ko "brush-stroke" style mein convert karta hai (MNIST mimicry).
  - **Auto-Inversion:** White background ko automatically black mein convert karta hai.
- **High Accuracy:** Trained on Google Colab GPU with **99.27% accuracy**.

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MNIST-Digit-Classifier.git
   cd MNIST-Digit-Classifier

2. Install dependencies:

pip install -r requirements.txt

## Usage

1. Draw a digit in any painting app (like Pinta) and save it as a .png file.
2. Run the prediction script:

python predict.py

3. Paste the path of your image and get the result!

## Model Architecture
Conv Layers: 2 Convolutional layers with Batch Normalization.
Pooling: MaxPool2d for downsampling.
Fully Connected: Linear layers with Dropout for final classification.
Optimizer: Adam Optimizer.
Loss Function: CrossEntropyLoss.

## Results
Training Accuracy: ~99.3%
Training Loss: ~0.002
