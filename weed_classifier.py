# weed_classifier.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

class WeedClassifier(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_classes=3):
        super(WeedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def load_model_and_mapping(model_path, split_info_path):
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)
    label_mapping = split_info['label_mapping']
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeedClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, idx_to_label, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_weed(image_path, model, idx_to_label, device):
    # Preprocess the image
    image_tensor = preprocess_image(image_path).to(device)
    feature_extractor = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1]).to(device)
    feature_extractor.eval()
    
    with torch.no_grad():
        # Extract features from the image
        features = feature_extractor(image_tensor).view(1, -1)
        
        # Pass features through the weed classification model
        outputs = model(features)
        
        # Apply softmax to get normalized probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top prediction and its confidence
        confidence, predicted = torch.max(probabilities, 1)

        # Process predictions for all classes
         # Process predictions for all classes
        predictions = [
            {'class': idx_to_label[i], 'confidence': float(prob) * 100}
            for i, prob in enumerate(probabilities.squeeze())
            if float(prob) * 100 > 0.01  # Filter out predictions with confidence <= 0.01%
        ]
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Return top prediction and all predictions
        return {
            'top_prediction': {
                'class': idx_to_label[predicted.item()],
                'confidence': float(confidence.item()) * 100
            },
            'all_predictions': predictions
        }

