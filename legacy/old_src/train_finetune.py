import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

def prepare_finetune_model(checkpoint_path="best_model.pth", num_classes=8, device="cuda"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, starting from random/pretrained base.")
        
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model.to(device)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning using device: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    train_dataset = datasets.ImageFolder("../train", transform=train_transform)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    val_subset.dataset.transform = eval_transform
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    model = prepare_finetune_model("best_model.pth", num_classes=len(train_dataset.classes), device=device)
    
    optimizer = AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-5}, 
        {"params": model.fc.parameters(), "lr": 1e-3}     
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    best_val_acc = 0.0
    

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters for Fine-Tuning: {trainable_params:,} out of {total_params:,}")
    
    print("\nStarting Fine-Tuning Training Loop...")
    for epoch in range(num_epochs):
     
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
       
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, "best_finetune_model.pth")
            print(f"--> Saved new best fine-tuned model with Val Acc: {val_acc:.4f}")

    print("\nFine-Tuning Complete! Best checkpoint saved as 'best_finetune_model.pth'.")