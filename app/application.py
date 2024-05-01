# Import libraries
import tensorflow as tf
import numpy as np
from flask import redirect, render_template, url_for, request, json, flash, Flask
from keras.models import load_model
from keras.preprocessing import image
import os
from datetime import datetime
from PIL import Image
from app import app

# Key for flash messages
app.secret_key = "qwerty098765421"

# Load serimpang model
serimpang_model = load_model('app/models/model_dir/serimpang256-densenet169-model1_2-acc99.78%.h5')

# Folder upload
UPLOAD_FOLDER = 'app/static/user_uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed filename extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load class dictionary from JSON
def load_class_mapping():
    with open('app/models/label_dir/serimpang256-model1_2-classes.json', 'r') as f:
        class_mapping = json.load(f)
    return class_mapping

# Render landing page
@app.route('/')
def index():
    return render_template('index.html')

# Render predict page
@app.route('/class-serimpang')
def class_serimpang():
    return render_template('class-serimpang.html')

# Render result page
@app.route('/result-class-serimpang', methods=['POST'])
def result_class_serimpang():
    # Handle file upload
    files = request.files.getlist('file')
    if not files:
        flash("Gambar belum dimasukkan. Mohon unggah gambar terlebih dahulu.")
        return render_template("class-serimpang.html")

    for file in files:
        if not allowed_file(file.filename):
            flash("File yang dipilih harus berformat jpg, jpeg, atau png.")
            return render_template("class-serimpang.html")

    # Save uploaded image
    filename = "temp_image.png"
    success = False
    for file in files:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        success = True

    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Preprocess and prepare image for prediction
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'app/static/user_uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    img.convert('RGB').save(predict_image_path, format="png")
    img.close()

    img = image.load_img(predict_image_path, target_size=(256, 256))
    x = image.img_to_array(img) / 255.0
    x = x.reshape(1, 256, 256, 3)
    images = np.array(x)

    # Load class mapping from JSON
    class_mapping = load_class_mapping()

    # Make prediction
    prediction_array_densenet169 = serimpang_model.predict(images)

    # Determine predicted class index and confidence using class mapping
    predicted_class_index = np.argmax(prediction_array_densenet169)
    predicted_class = class_mapping.get(str(predicted_class_index))  # Use string key for dictionary lookup

    # Prepare response data
    confidence = '{:.2%}'.format(np.max(prediction_array_densenet169))

    return render_template("class-serimpang-predict.html", img_path=img_url,
                           prediction=predicted_class, confidence=confidence)

# Render about page
@app.route('/about')
def about():
    return render_template('about.html')