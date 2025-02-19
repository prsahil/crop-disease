# agriapp/views.py
from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import io

# Load models
disease_model = tf.keras.models.load_model("models/disease_model.h5")
irrigation_model = joblib.load("models/irrigation_model.pkl")

# Disease Classes
classes = ["Healthy", "Disease A", "Disease B", "Disease C"]

def home(request):
    return render(request, "index.html")

# Disease Detection API
def detect_disease(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        img = Image.open(image).resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = disease_model.predict(img_array)
        disease = classes[np.argmax(prediction)]
        return JsonResponse({"disease": disease, "suggestion": "Use organic pesticide"})

    return JsonResponse({"error": "Invalid request"})

# Irrigation Prediction API
def predict_irrigation(request):
    if request.method == "POST":
        temp = float(request.POST.get("temperature"))
        humidity = float(request.POST.get("humidity"))
        soil_moisture = float(request.POST.get("soil_moisture"))

        prediction = irrigation_model.predict([[temp, humidity, soil_moisture]])
        return JsonResponse({"irrigation_time": prediction[0]})

    return JsonResponse({"error": "Invalid request"})
