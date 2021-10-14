from fastapi import APIRouter, File, UploadFile
from model import ImagePrediction
from starlette.responses import JSONResponse
from PIL import Image
import io

router = APIRouter()
img_height, img_width = 32, 32


def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB").resize((img_height, img_width))
    return image


@router.post("/predict_image")
async def predict_class(file: UploadFile = File(...)):
    classifier = ImagePrediction()
    image = read_imagefile(await file.read())
    return JSONResponse(classifier.predict(image))
