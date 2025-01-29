import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm

# Paths
MODEL_PATH = r"C:\sreejha_project\project\models\improved_classifier.pth"
LABEL_MAPPING_PATH = r"C:\sreejha_project\project\models\label_mapping.npy"

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.model.classifier = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

# Classifier
class ImprovedClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(ImprovedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    # Load label mapping
    label_mapping = np.load(LABEL_MAPPING_PATH, allow_pickle=True).item()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    
    classifier = ImprovedClassifier(
        input_features=feature_extractor.model.num_features, 
        num_classes=len(label_mapping)
    ).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        outputs = classifier(features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get predicted label
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_mapping[predicted_class]
    
    return predicted_label, confidence

if __name__ == "__main__":
    # Example usage - replace with your image path
    image_path = r"c:\Users\Sreejha\Desktop\3.jpg"
    
    predicted_label, confidence = predict(image_path)
    print(f"\nImage: {image_path}")
    print(f"Predicted Disease: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    
    
    