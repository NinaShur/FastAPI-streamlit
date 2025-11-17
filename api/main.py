import logging
import random
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.model_func import (
    load_pt_model,
    load_sklearn_model
)

logger = logging.getLogger('uvicorn.info')

# КЛАССЫ ДЛЯ ДЕТЕКЦИИ ОБЪЕКТОВ
class Detection(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    bbox: list[float]

class ImageResponse(BaseModel):
    detections: list[Detection]
    total_objects: int

# КЛАССЫ ДЛЯ ТЕКСТА
class TextInput(BaseModel):
    text: str

class TextResponse(BaseModel):
    label: str
    prob: float

# КЛАССЫ ДЛЯ ТАБЛИЧНЫХ ДАННЫХ (ДЕМО)
class TableInput(BaseModel):
    feature1: float
    feature2: float

class TableOutput(BaseModel):
    prediction: float

pt_model = None
sk_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pt_model, sk_model
    pt_model = load_pt_model()
    logger.info('YOLO model loaded')
    
    # РАСКОММЕНТИРУЙТЕ ЭТУ СТРОКУ!
    sk_model = load_sklearn_model()
    logger.info('NLP model loaded')
    
    yield
    del pt_model, sk_model

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Hello FastAPI with YOLO and NLP!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    image = PIL.Image.open(file.file)
    results = pt_model(image)
    result = results[0]
    
    detections = []
    for box in result.boxes:
        detections.append({
            'class_name': result.names[int(box.cls)],
            'class_index': int(box.cls),
            'confidence': float(box.conf),
            'bbox': box.xyxy[0].tolist()
        })
    
    return {
        'detections': detections,
        'total_objects': len(detections)
    }

@app.post('/clf_text')
def clf_text(data: TextInput):
    # УБЕРИТЕ ПРОВЕРКУ sk_model is None - теперь модель всегда загружена
    prediction = sk_model(data.text)[0]
    
    response = TextResponse(
        label=prediction['label'],
        prob=prediction['score']
    )
    return response

@app.post('/clf_table')
def predict(x: TableInput):
    # Демо-функция
    prediction = random.randint(0, 1)
    result = TableOutput(prediction=prediction)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
