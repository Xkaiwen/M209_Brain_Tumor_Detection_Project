from flask import Flask, request, render_template, make_response, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from datetime import datetime
import base64
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
model = load_model('/Users/kaiwenxue/Desktop/Project/models/brain_tumor_detection_cnnmodel1_2.h5')

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def img_pred(img_path):
    """Predict tumor type from the image."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Error: Unable to read the image at {img_path}.")
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 149.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)[0]

def encode_image_to_base64(img):
    """Convert an image to Base64 format for embedding in HTML."""
    success, buffer = cv2.imencode('.jpg', img)
    if success:
        return base64.b64encode(buffer).decode('utf-8')
    return None

def preprocess(img_path):
    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise ValueError(f"Error: Unable to process the image at {img_path}.")
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    #Salt&Pepper Noise Removal
    median_filtered = cv2.medianBlur(gray_img, 5)

    #Edge Detection
    img_sobelx = cv2.Sobel(median_filtered, cv2.CV_8U, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(median_filtered, cv2.CV_8U, 0, 1, ksize=3)
    img_sobel = img_sobelx + img_sobely

    #Threshold
    _, thresh = cv2.threshold(img_sobel, 50, 255, cv2.THRESH_BINARY)

    images = {
        'original': encode_image_to_base64(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)),
        'salt and pepper noise removal': encode_image_to_base64(median_filtered),
        'edge_detected': encode_image_to_base64(img_sobel),
        'threshold': encode_image_to_base64(thresh),
    }
    return images

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect', methods=['GET'])
def detect():
    """Generate unique session token to attach to file upload form."""
    token = uuid.uuid4().hex
    return render_template('detect.html', token=token)

@app.route('/result', methods=['POST'])
def result():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        processed_images = preprocess(file_path)
        prediction = img_pred(file_path)
        tumor_types = {
            0: ("Malignant tumor detected.", "Type: Glioma."),
            1: ("Benign tumor detected.", "Type: Meningioma."),
            2: ("No tumor detected.", ""),
            3: ("Benign tumor detected.", "Type: Pituitary adenoma.")
        }
        message, tumor_type = tumor_types.get(prediction, ("Error in prediction.", ""))

        response = make_response(render_template(
            'result.html', prediction=prediction, message=message, tumor_type=tumor_type, images=processed_images
        ))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return "Invalid file type. Please upload a valid image.", 400

if __name__ == '__main__':
    app.run(debug=True)