import torch
from torchvision.models import resnet18
import torchvision.transforms as T
import json
import joblib
import os
from ultralytics import YOLO
from transformers import pipeline
import numpy as np

# def class_id_to_label(i):
#     '''
#     Input int: class index
#     Returns class name
#     '''

#     labels = load_classes()
#     return labels[i]

def load_pt_model():
    model = YOLO('yolov8n.pt')
    return model

# # функция для анализа тональности текста
# def load_sklearn_model():
#     # Более легкая и быстрая модель
#     return pipeline("sentiment-analysis", 
#                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")



# # потом попробовать эту! долго грузит стримлит и ничего
# def load_sklearn_model():
#     from transformers import pipeline
#     try:
#         # Маленькая модель которая точно скачается
#         return pipeline("text-classification", 
#                        model="finiteautomata/bertweet-base-sentiment-analysis",
#                        max_length=512)
#     except Exception as e:
#         print(f"Ошибка: {e}")
#         # Заглушка
#         class SimpleModel:
#             def __call__(self, text):
#                 return [{'label': 'POSITIVE', 'score': 0.9}]
#         return SimpleModel()
    
# def load_sklearn_model():
#     # Умная заглушка - РАБОТАЕТ БЕЗ ИНТЕРНЕТА
#     class SmartSentiment:
#         def __call__(self, text):
#             text_lower = text.lower()
#             # Логика на ключевых словах
#             if any(word in text_lower for word in ['love', 'good', 'great', 'awesome']):
#                 return [{'label': 'POSITIVE', 'score': 0.92}]
#             elif any(word in text_lower for word in ['hate', 'bad', 'terrible']):
#                 return [{'label': 'NEGATIVE', 'score': 0.88}]
#             else:
#                 return [{'label': 'NEUTRAL', 'score': 0.75}]
#     return SmartSentiment()

def load_sklearn_model():
    # УМНАЯ ЗАГЛУШКА С ВАШИМИ 5 КЛАССАМИ
    class SmartTopicClassifier:
        def __call__(self, text):
            text_lower = text.lower()
            
            # Логика для ваших 5 тем
            if any(word in text_lower for word in ['крипт', 'биткоин', 'блокчейн', 'технолог', 'программ', 'ai', 'ии']):
                return [{'label': 'технологии', 'score': 0.95}]
            elif any(word in text_lower for word in ['искусств', 'арт', 'живопис', 'музык', 'театр', 'кино']):
                return [{'label': 'искусство', 'score': 0.92}]
            elif any(word in text_lower for word in ['образован', 'учеба', 'курс', 'обучен', 'познавательн', 'наук']):
                return [{'label': 'образование_познавательное', 'score': 0.90}]
            elif any(word in text_lower for word in ['маркетинг', 'реклам', 'продаж', 'бизнес', 'торговл']):
                return [{'label': 'маркетинг', 'score': 0.88}]
            elif any(word in text_lower for word in ['здоровь', 'спорт', 'медицин', 'врач', 'болезн', 'лечен']):
                return [{'label': 'здоровье_медицина', 'score': 0.93}]
            else:
                return [{'label': 'технологии', 'score': 0.8}]  # по умолчанию
            
    return SmartTopicClassifier()

def transform_image(img):
    # YOLO сам делает ресайз и нормализацию
    return np.array(img)

