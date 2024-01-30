from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Importing the model
classifier_model = load_model("classificationModel.keras")

# Loading and preprocessing an image to test
img_path = "image path" # Get image from testing folder and use it HERE
testImg = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(testImg)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 # Normalizing the image

# Making the prediction
predictions = classifier_model.predict(img_array)

# Get the predicted class
predicted_severity = np.argmax(predictions)
print(f"Predicted Severity: {predicted_severity}")
