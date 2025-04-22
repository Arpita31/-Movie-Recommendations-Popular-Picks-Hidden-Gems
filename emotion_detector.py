import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
MAX_LENGTH = 128

class AdvancedEmotionDetector:
  def __init__(self, model_path="./emotion_model_advanced"):
    try:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.model = AutoModelForSequenceClassification.from_pretrained(
          model_path,
          device_map='auto',
          torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
      )
      self.emotions = EMOTIONS
      try:
          self.explanations = pd.read_csv(f"{model_path}/explanations.csv")
      except:
          self.explanations = None

    except Exception as e:
      print("Initiating pre-trained model...")
      model_name = "j-hartmann/emotion-english-distilroberta-base"
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
      self.emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
      self.explanations = None

  def detect(self, text):
    inputs = self.tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    ).to(self.device)

    with torch.no_grad():
        outputs = self.model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    probabilities = {emotion: prob.item() for emotion, prob in zip(self.emotions, probs[0])}
    sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    explanation = None
    if self.explanations is not None:
        try:
            emotion = self.emotions[pred_class]
            explanation_row = self.explanations[self.explanations['emotion'] == emotion].sample(1)
            explanation = explanation_row['explanation'].values[0]
            explanation = explanation.replace('[TEXT]', text)
        except:
            explanation = None

    return {
        'emotion': self.emotions[pred_class],
        'confidence': probs[0][pred_class].item(),
        'emotion_ranking': sorted_emotions,
        'explanation': explanation
    }