import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm
import sys

classes = ["Cyst", "Normal", "Stone", "Tumor"]

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridCNNViT, self).__init__()
        # CNN backbone
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()  # remove last FC

        # Vision Transformer backbone
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.vit.head = nn.Identity()

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(512+768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        out = self.classifier(combined)
        return out

def predict_image(image_path, model_path="model/hybrid_cnn_vit.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridCNNViT(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = output.max(1)
        predicted_class = classes[pred.item()]

    return predicted_class

if __name__ == "__main__":
    image_path = r"CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\Tumor\Tumor- (9).jpg" 

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    result = predict_image(image_path, model_path="model/hybrid_cnn_vit.pth")
    print(f"\nPredicted Class: {result}")
