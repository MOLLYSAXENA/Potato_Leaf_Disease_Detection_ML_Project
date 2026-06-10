import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 1. Setup device mapping (Spaces use CPU by default)
device = torch.device("cpu")

# 2. Define the production validation transform pipeline
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Rebuild the ResNet-18 Architecture
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: Diseased vs Healthy

# 4. Load your saved weights safely onto the CPU
model.load_state_dict(torch.load("production_potato_resnet18.pth", map_location=device))
model.eval()

# Explicitly map the class names
classes_map = ["Diseased", "Healthy"]

# 5. Define the prediction function
def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocess image
    img_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
    return f"Diagnosis: {classes_map[predicted.item()]}"

# 6. Build the web interface layout
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="Potato Leaf Disease Diagnostics Portal",
    description="Upload a photo of a potato leaf to instantly identify if it is healthy or affected by Early Blight."
)

if __name__ == "__main__":
    interface.launch()
