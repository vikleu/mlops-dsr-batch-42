import torch
import io 
from pydantic import BaseModel 
from fastapi import FastAPI, File, UploadFile, Depends
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms
from app.model import load_model, load_transforms
from PIL import Image

CATEGORIES = ['fresh_apple', 'fresh_banana', 'fresh_orange', 
              'rotten_apple', 'rotten_banana', 'rotten_orange']

# This is a  'data model' for the output of the fruit classifier  
class Result(BaseModel):
    category: str # predicted label for the image
    confidence: float # confidence score for the prediction

app = FastAPI()


@app.get("/")
def read_root():
    return {'message': 
            'Welcome to the Fruit Classifier API, call the endpoint through /predict to classify images'}

@app.post("/predict", response_model=Result)
async def predict(
    input_image: UploadFile = File(...),
    model: ResNet = Depends(load_model),
    transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read())).convert('RGB')

    #non_sense_result = Result(category=f'{image.mode}', confidence=0.0)

    image = transforms(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        predicted_category = CATEGORIES[predicted_class]

        result = Result(category=predicted_category, 
                confidence=confidence.item())

    return result 