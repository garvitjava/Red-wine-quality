import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data as mentioned in the previous response
# Load your red wine dataset (replace 'your_dataset.csv' with your data file)
data = pd.read_csv('winequality-red.csv')

# Separate features and labels
X = data.drop('quality', axis=1).values
y = data['quality'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert quality labels to the range [0, 9]
y = y - 1  # Subtract 1 to make it 0-based (0 for worst, 9 for best)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='linear')
])

# Compile the Model with logits=True
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Get the logits (raw scores) for each class
logits = model.predict(X_test)

# Apply softmax to obtain class probabilities
probabilities = tf.nn.softmax(logits)
