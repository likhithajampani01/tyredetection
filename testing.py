import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load condition model
condition_model = tf.keras.models.load_model(r'C:\Users\HP\PycharmProjects\tyredetection\tyre_condition_model.h5')

# Load pressure-lifespan model
pressure_lifespan_model = tf.keras.models.load_model(r'C:\Users\HP\PycharmProjects\tyredetection\tyre_pressure_lifespan_model.h5')

# Prepare Image
image_path = r'C:\Users\HP\PycharmProjects\tyredetection\fixed_splittingdata\val\good\good (4).jpg'
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict condition
prediction = condition_model.predict(img_array)

if prediction[0][0] >= 0.5:
    condition = "GOOD TYRE"
    condition_value = 1
else:
    condition = "DEFECTIVE TYRE"
    condition_value = 0

print(f"🧾 Prediction: {condition}")

# Predict Pressure & Lifespan
# Example input features (replace with real sensor values or user input!)
age_in_years = 2
tread_depth_mm = 4.5
ambient_temperature_C = 30

input_features = np.array([[condition_value, age_in_years, tread_depth_mm, ambient_temperature_C]])
predicted_values = pressure_lifespan_model.predict(input_features)

predicted_pressure = predicted_values[0][0]
predicted_lifespan = predicted_values[0][1]

print(f"🔧 Predicted Pressure: {predicted_pressure:.2f} PSI")
print(f"🛞 Predicted Lifespan: {predicted_lifespan:.0f} km")
