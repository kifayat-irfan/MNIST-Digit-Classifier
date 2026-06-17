import torch
import torch.nn as nn
import torch.optim as optim
import config
from model import DigitCNN
from dataset import get_loaders
import os
import matplotlib.pyplot as plt

def main():
    os.makedirs('weights', exist_ok=True)
    train_loader, test_loader = get_loaders()
    model = DigitCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Trackers
    epoch_list, loss_list, accuracy_list = [], [], []
    best_acc = 0.0
    start_epoch = 0

    # --- RESUME LOGIC ---
    checkpoint_path = 'weights/checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(" Found checkpoint! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resuming from Epoch {start_epoch + 1}")

    try:
        print(f" Training started. Press Ctrl+C to stop and save graph!")
        for epoch in range(start_epoch, config.EPOCHS):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = total_loss / len(train_loader)
            acc = 100 * correct / total
            
            epoch_list.append(epoch + 1)
            loss_list.append(avg_loss)
            accuracy_list.append(acc)
            
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")

            # --- BEST MODEL SAVING ---
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f" New Best Model Saved! Acc: {best_acc:.2f}%")

            # --- SAVE CHECKPOINT (For Resume) ---
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)

    except KeyboardInterrupt:
        print("\n Training stopped manually by user!")
    
    finally:
        # --- FINAL GRAPH (Will run whether training ends or is stopped) ---
        if epoch_list:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_list, loss_list, color='red')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(epoch_list, accuracy_list, color='blue')
            plt.title('Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            
            plt.tight_layout()
            plt.savefig('training_results.png')
            plt.show()
            print("\n Graph saved as training_results.png")
        else:
            print("No data to plot.")

if __name__ == "__main__":
    main()
