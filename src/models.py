import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=8):
    """
    لود کردن ResNet-18، فریز کردن لایه‌های پایه و تغییر لایه آخر (Head)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
   
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
   
    my_model = create_model(num_classes=8)
    my_model = my_model.to(device)
    

    print("--- Active layers for training ---")
    active_layers = 0
    for name, param in my_model.named_parameters():
        if param.requires_grad:
            print(f"Training: {name}")
            active_layers += 1
            
    if active_layers == 2:
        print("\n✅ Model successfully frozen, only the last layer is ready for training.")


class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleTrafficCNN, self).__init__()
        
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    
    model = SimpleTrafficCNN(num_classes=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Simple Custom CNN Total Parameters: {total_params:,}")



class TrafficNetProficient(nn.Module):
    def __init__(self, num_classes=8):
        super(TrafficNetProficient, self).__init__()
        
    
        self.feature_extractor = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = TrafficNetProficient(num_classes=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Proficient Custom CNN Total Parameters: {total_params:,}")