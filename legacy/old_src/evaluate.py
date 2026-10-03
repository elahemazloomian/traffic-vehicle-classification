import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import json
import numpy as np

def load_trained_model(checkpoint_path="../best_model.pth", num_classes=8, device="cuda"):

    # بارگذاری مدل ResNet18 مطابق ساختار آموزش‌داده‌شده
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, class_names, transform, device="cuda", threshold=0.70):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, dim=0)
        
    pred_class = class_names[pred_idx.item()]
    confidence = conf.item()
    
    prob_dict = {class_names[i]: round(probabilities[i].item(), 4) for i in range(len(class_names))}
    
    result = {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "needs_review": confidence < threshold
    }
    return result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}")
    
    # تنظیمات ترنسفرم برای ارزیابی
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder("../test", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    model = load_trained_model("best_model.pth", num_classes=len(test_dataset.classes), device=device)
    
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # محاسبه متریک‌های ارزیابی
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    macro_p = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    print("\n--- Evaluation Results on Test Set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=test_dataset.classes))