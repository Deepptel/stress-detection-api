from fastapi import FastAPI
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

app = FastAPI()

# Load Model & Tokenizer
model = TFBertForSequenceClassification.from_pretrained("stress_detection_model")
tokenizer = BertTokenizer.from_pretrained("stress_detection_model")

@app.post("/predict")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=160)
    outputs = model(inputs)[0]
    prediction = tf.argmax(outputs, axis=1).numpy()[0]
    return {"stress": bool(prediction)}

# Run the API using this command: uvicorn app:app --host 0.0.0.0 --port 8000
# python -m uvicorn app:app --host 0.0.0.0 --port 8000