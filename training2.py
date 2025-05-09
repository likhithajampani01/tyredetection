import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample Data: [condition_value, age_in_years, tread_depth_mm, ambient_temperature_C]
X = np.array([
    [1, 3, 4.0, 35],
    [0, 1, 6.0, 25],
    [1, 5, 2.5, 38],
    [0, 2, 5.5, 30],
    [1, 4, 3.0, 33],
    [0, 1, 7.0, 22]
])

# Target: [pressure in psi, lifespan in km]
Y = np.array([
    [28, 25000],
    [35, 45000],
    [25, 20000],
    [33, 40000],
    [27, 22000],
    [36, 47000]
])

# Build regression model
pressure_lifespan_model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(2)  # pressure and lifespan
])

pressure_lifespan_model.compile(optimizer='adam', loss='mse')

# Train model
pressure_lifespan_model.fit(X, Y, epochs=300, verbose=0)

# Save model
pressure_lifespan_model.save(r'C:\Users\HP\PycharmProjects\tyredetection\tyre_pressure_lifespan_model.h5')
print("✅ Pressure & Lifespan model trained and saved.")
