from fastapi import FastAPI, UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from .model import Inference

app = FastAPI()
labels = ['CAT', 'DOG']

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"]
)

@app.post("/Classification")
async def classifier(data:UploadFile = File(...)):
    file_location = f"temp_{data.filename}"

    with open(file_location,"wb") as buffer:
        buffer.write(await data.read())

    pred_class,conf = Inference(file_location)

    return{
        "pred_label":labels[pred_class],
        "conf":conf*100
    }