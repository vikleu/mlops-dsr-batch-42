import wandb
import torch
import os 
from torchvision.models import ResNet, resnet18
from torchvision.transforms import v2 as transforms
from loadotenv import load_env
import torch.nn as nn 
from pathlib import Path

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_FILE_NAME = "best_model.pth"

load_env()  # load environment variables from .env file
# comment load_env() when doing the GCP deployment
wandb_api_key = os.environ.get("WANDB_API_KEY")
model_path = os.environ.get("MODEL_PATH")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    #print("Logged into weights & biases")

def download_artifact():
    artifact = wandb.Api().artifact(model_path,type="model")
    artifact.download(MODELS_DIR)

def get_raw_model() -> ResNet:
    """We create a ResNet model with the same architecture as the once we trained."""
    architecture = resnet18(weights=None)  # we set weights=None to create a model without pretrained weights
    architecture.fc = nn.Sequential(
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,6)
    )

    return architecture 

# you should see best_model.pth inside of the models directory
# download_artifact()
def load_model() -> ResNet:
    download_artifact()
    model = get_raw_model()
    ...

    # Get trained model weights from the models directory
    mode_state_dict_path = Path(MODELS_DIR) / MODEL_FILE_NAME
    model_state_dict = torch.load(os.path.join(MODELS_DIR, MODEL_FILE_NAME), 
                                  map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict, strict=True)
    model.eval() # Set the model to evaluation mode
    # Now we have the model loaded with the trained weights
    return model


def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229 , 0.224, 0.225])
    ])