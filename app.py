from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

app1 = Flask(__name__)

models = {
    "model1": "emobobas/celebrity_deepfake_detection",
    "model2": "imdaisylee/test_model"
}

@app1.route('/')
def home():
    return "Deepfake vs Real Image Detection API"

@app1.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file or model part'})
    
    file = request.files['file']
    model_key = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if model_key not in models:
        return jsonify({'error': 'Invalid model selected'})

    try:
        # Load the model
        processor = AutoImageProcessor.from_pretrained(models[model_key])
        model = AutoModelForImageClassification.from_pretrained(models[model_key])

        # Open the image file
        image = Image.open(file).convert('RGB')
        
        # Preprocess the image
        processor = models[model_key]["processor"]
        model = models[model_key]["model"]
        inputs = processor(images=image, return_tensors="pt")   
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the predicted class and confidence
        predicted_class_idx = outputs.logits.argmax(-1).item()
        confidence_score = torch.softmax(outputs.logits, dim=-1)[0, predicted_class_idx].item()
        predicted_class = model.config.id2label[predicted_class_idx]
        
        # Return the prediction and confidence
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence_score})
    
    
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app1.run(debug=True, host='0.0.0.0', port=port)
