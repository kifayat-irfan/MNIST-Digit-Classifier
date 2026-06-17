import torch
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from model import DigitCNN
import config
import numpy as np
import os

def predict_image(img_path):
    # 1. CLEAN THE PATH
    img_path = img_path.strip().replace('"', '').replace("'", "")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f" File not found at: {img_path}")

    # 2. Load Model
    model = DigitCNN().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    # 3. Open and Convert to Grayscale
    image = Image.open(img_path).convert('L')

    # 4. AUTO-INVERSION
    img_array = np.array(image)
    if np.mean(img_array) > 127: 
        image = ImageOps.invert(image)
    
    # --- NEW: MAKE LINES THICKER (MNIST Style) ---
    # GaussianBlur spreads the lines a bit, making them look like brush strokes.
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # ---------------------------------------------

    # 5. AUTOMATIC CROPPING
    img_array = np.array(image)
    coords = np.column_stack(np.where(img_array > 50)) # Threshold 50 for blur
    if coords.size == 0:
        return None, 0

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    digit_crop = image.crop((x_min, y_min, x_max, y_max))
    
    # Make it Square
    w, h = digit_crop.size
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    final_image = ImageOps.expand(digit_crop, border=(pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h), fill=0)

    # 6. Preprocess for Model
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --- DEBUG: Save what AI sees ---
    # This image will be saved so you can see what the AI is seeing
    debug_img = final_image.resize((28, 28))
    debug_img.save("debug_processed.png")
    # -------------------------------

    image_tensor = transform(final_image).unsqueeze(0).to(config.DEVICE)

    # 7. Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

if __name__ == "__main__":
    print("\n---  MNIST Digit Predictor (Optimized Version) ---")
    path = input("Paste the image path: ")
    
    try:
        result, conf = predict_image(path)
        if result is None:
            print(" Error: No digit detected!")
        else:
            print(f"\n AI Prediction: The digit is {result}")
            print(f" Confidence: {conf*100:.2f}%")
            print(f"  Check 'debug_processed.png' to see what the AI saw!")
        print("-----------------------------------")
    except Exception as e:
        print(f" An error occurred: {e}")
