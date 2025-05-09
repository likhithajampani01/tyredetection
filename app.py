from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# Load your models
condition_model = tf.keras.models.load_model(r'C:\Users\HP\PycharmProjects\tyredetection\tyre_condition_model.h5')
pressure_lifespan_model = tf.keras.models.load_model(r'C:\Users\HP\PycharmProjects\tyredetection\tyre_pressure_lifespan_model.h5')

@app.route('/')
def upload_page():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'tyre_image' not in request.files:
        return 'No image uploaded!', 400

    file = request.files['tyre_image']
    if file.filename == '':
        return 'No selected image!', 400
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict tyre condition
    prediction = condition_model.predict(img_array)
    if prediction[0][0] >= 0.5:
        condition = "GOOD TYRE"
        condition_value = 1
    else:
        condition = "DEFECTIVE TYRE"
        condition_value = 0

    # Now pass the condition to your pressure-lifespan model
    # Example: Default scenario values (can be adjusted to your business logic)
    age_in_years = 2
    tread_depth_mm = 4.5
    ambient_temperature_C = 30

    input_features = np.array([[condition_value, age_in_years, tread_depth_mm, ambient_temperature_C]])
    predicted_values = pressure_lifespan_model.predict(input_features)

    predicted_pressure = round(predicted_values[0][0], 2)
    predicted_lifespan = round(predicted_values[0][1])

    return render_template(
        'result.html',
        condition=condition,
        pressure=predicted_pressure,
        lifespan=predicted_lifespan
    )
if __name__ == '__main__':
    app.run(debug=True)
