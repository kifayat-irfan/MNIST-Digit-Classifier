# ✏️ MNIST Digit Classifier (CNN)

I built an end-to-end handwritten digit recognizer in **PyTorch**: a custom CNN trained on the MNIST dataset, plus a prediction script that reads digits I draw in any painting app.

## ✨ Features
- **Custom CNN:** 2 conv layers (32 → 64 filters) with BatchNorm, MaxPool, a 128-unit fully connected layer with BatchNorm + Dropout (0.3), and CrossEntropyLoss with Adam. (See `model.py`.)
- **Smart preprocessing** in `predict.py`:
  - **Auto-inversion** — white-background drawings are flipped to MNIST's black-background style.
  - **Gaussian blur** — thickens sharp lines so they look like brush strokes.
  - **Automatic bounding-box cropping** — crops the digit and pads it square so it isn't stretched.
- **Debug view:** every prediction saves `debug_processed.png` so I can see exactly what the model was shown.

## 📈 Results
- **Best test accuracy: ≈99.3%** — see `training_results.png` for the loss/accuracy curves.
- Trained weights are included: `weights/mnist_cnn.pth`.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kifayat-irfan/MNIST-Digit-Classifier.git
   cd MNIST-Digit-Classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. Train (optional — weights are already included):
   ```bash
   python train.py
   ```
   Training auto-saves the best weights, supports resume-from-checkpoint, and plots `training_results.png`. CUDA is used when available (set in `config.py`).
2. Draw a digit in any painting app (like Pinta) and save it as a `.png`.
3. Predict:
   ```bash
   python predict.py
   ```
   Paste the image path and get the digit + confidence.

## 📂 Files
- `model.py` — CNN architecture
- `train.py` — training loop with checkpointing and best-model saving
- `predict.py` — prediction + preprocessing
- `dataset.py` — MNIST loaders
- `config.py` — hyperparameters (50 epochs, batch 64, lr 0.001)
- `weights/mnist_cnn.pth` — trained weights
- `test_images/` — sample hand-drawn digits
- `training_results.png` — loss/accuracy curves
