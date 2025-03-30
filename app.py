# Install required packages locally (run these in your terminal):
# pip install flask tensorflow==2.17.0 opencv-python pillow numpy

# Import libraries
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
print(tf.__version__)  # Should be 2.17.0 (or your installed version)
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Define the focal loss function with serialization decorator
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=None, gamma=1.0):
    if alpha is None:
        alpha = [0.1347, 0.0857, 0.0401, 0.3832, 0.0396, 0.0066, 0.3102]
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.convert_to_tensor(alpha, dtype=tf.float32) * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(weight * ce)
    return loss

# Load your trained model
model_path = "final_optimized_model.keras"  # Adjust path if needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

try:
    model = tf.keras.models.load_model(model_path, custom_objects={'loss': focal_loss})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Define label encoder classes
label_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
label_name = {
    "akiec": {"name": "Actinic Keratoses and Intraepithelial Carcinoma (Bowen’s Disease)", "Desc": "A type of precancerous lesion or early-stage squamous cell carcinoma."}, 
    "bcc": {"name": "Basal Cell Carcinoma", "Desc": "The most common type of skin cancer, usually slow-growing and rarely metastatic."}, 
    "bkl": {"name": "Benign Keratosis-Like Lesions", "Desc": "Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses, which are benign skin conditions."}, 
    "df": {"name": "Dermatofibroma", "Desc": "A benign skin lesion, usually firm and small, often occurring on the lower extremities."}, 
    "mel": {"name": "Melanoma", "Desc": "A malignant skin tumor originating from melanocytes, often aggressive and life-threatening if not detected early."}, 
    "vasc": {"name": "Vascular Lesions", "Desc": "Includes hemangiomas, angiokeratomas, pyogenic granulomas, and other vascular anomalies."},
    "nv": {"name": "Melanocytic Nevi (Moles)", "Desc": "Common benign moles that originate from melanocytes"}
}

# Validate the image
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the file is a valid image
        return True
    except Exception as e:
        print(f"Invalid image: {e}")
        return False

# Detect if the image contains a lesion
def detect_lesion(image_path):
    print("detect_lesion started: " + image_path)
    image = preprocess_image(image_path)
    predictions = model.predict(image)[0]  # Get predictions for the image
    print("Prediction values: " + str(predictions))
    max_prob = np.max(predictions)
    threshold = 0.4  # Adjust threshold based on validation
    print("Max probability: " + str(max_prob))
    print("Threshold: " + str(threshold))
    is_lesion = max_prob > threshold
    return is_lesion, max_prob, image, predictions
'''
# Preprocess the image for model prediction
def preprocess_image(image_path, target_size=(300, 300)):
    print("Preprocessing started")
    if not os.path.exists(image_path):
        raise ValueError(f"File not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Invalid image file: {image_path}")
    
    # Convert from BGR to RGB and resize
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, target_size)
    # Use EfficientNet's preprocessing (matches training)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print("Preprocessing completed")
    return image
'''
def preprocess_image(image_path, target_size=(300, 300)):
    print("Preprocessing started")
    if not os.path.exists(image_path):
        raise ValueError(f"File not found: {image_path}")
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    # Debug: Print shape and min/max values
    img_np = img.numpy()
    print("Image shape:", img_np.shape)
    print("Min, Max:", img_np.min(), img_np.max())
    img_np = np.expand_dims(img_np, axis=0)
    print("Preprocessing completed")
    return img_np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    print("Detecting...")
    try:
        if 'file' not in request.files:
            print("Error: No file uploaded")
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files['file']
        if file.filename == '':
            print("Error: No file selected")
            return jsonify({"error": "No file selected!"}), 400

        # Save the uploaded image temporarily
        file_path = "temp2.jpg"
        file.save(file_path)

        # Validate the image
        if not is_valid_image(file_path):
            os.remove(file_path)
            print("Uploaded file is not a valid image.")
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        # Detect lesion in the image
        is_lesion, max_prob, image, predictions = detect_lesion(file_path)
        print(f"Lesion detected: {is_lesion}")
        print(max_prob)

        if is_lesion:
            print("Processing prediction...")
            predicted_index = int(np.argmax(predictions))
            predicted_class = list(label_name.keys())[predicted_index]
            predicted_name = label_name[predicted_class]["name"]
            predicted_desc = label_name[predicted_class]["Desc"]
            max_conf = float(max_prob)
        else:
            # If no lesion is detected, return default values
            predicted_class = "none"
            predicted_name = "No lesion detected"
            predicted_desc = ""
            max_conf = 0.0

        # Clean up temporary file
        os.remove(file_path)

        return jsonify({
            "lesion_detected": bool(is_lesion),
            "max_probability": float(round(max_prob, 2)),
            "class": predicted_class,
            "name": predicted_name,
            "classDesc": predicted_desc,
            "max_confidence": max_conf
        })
    except Exception as e:
        print(f"Error in /detect endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)