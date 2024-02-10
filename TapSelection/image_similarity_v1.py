import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Load the pretrained model
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')

#print(model)

# Set the model to evaluation mode
model.eval()

# Define a function to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Load and preprocess two images
image1 = preprocess_image('bottle_1_wb.jpg')
image2 = preprocess_image('bottle_1_wb_rotated.jpg')

# Get the model predictions for the two images
with torch.no_grad():
    output1 = model(image1)
    output2 = model(image2)

# Using the outputs as the embeddings 
embedding1 = output1
embedding2 = output2

# Calculate the L2 norm (Euclidean distance) between the two embeddings
similarity_score = F.pairwise_distance(embedding1, embedding2)

print("Dissimilarity Score (L2 norm):", similarity_score.item())
