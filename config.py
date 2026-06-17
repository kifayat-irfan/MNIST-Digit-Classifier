import torch

# Training Settings
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_PATH = './data'
MODEL_SAVE_PATH = './weights/mnist_cnn.pth'
