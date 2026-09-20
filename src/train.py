import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import get_data_loaders
from models import create_model

def train_model(num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training running on: {device}")
    if device.type == 'cuda':
        print(f" GPU in use: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = get_data_loaders(data_dir="../dataset_cleaned")
    model = create_model(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=0.001)


    best_metrics = {
        'epoch': 0,
        'train_acc': 0.0,
        'train_loss': 0.0,
        'val_acc': 0.0,
        'val_loss': 0.0
    }


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 25)
        start_time = time.time()

  
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects.double() / len(train_loader.dataset)


        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

 
        print(f"Train -> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
        print(f"Val   -> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")

   
        if epoch_val_acc > best_metrics['val_acc']:
       
            best_metrics['epoch'] = epoch + 1
            best_metrics['val_acc'] = epoch_val_acc
            best_metrics['val_loss'] = epoch_val_loss
            best_metrics['train_acc'] = epoch_train_acc
            best_metrics['train_loss'] = epoch_train_loss
            
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model weights saved successfully!")
            
        print(f"Duration: {time.time() - start_time:.0f}s")

  
    print(f"\n Training finished! Best metrics were achieved in Epoch {best_metrics['epoch']}:")
    print(f" Best Val   -> Loss: {best_metrics['val_loss']:.4f} | Acc: {best_metrics['val_acc']:.4f}")
    print(f" Same Epoch Train -> Loss: {best_metrics['train_loss']:.4f} | Acc: {best_metrics['train_acc']:.4f}")

if __name__ == "__main__":
    train_model(num_epochs=5)