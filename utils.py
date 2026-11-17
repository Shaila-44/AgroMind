import cv2
import numpy as np

def preprocess_image(image_bytes: bytes, target_size=(128, 128)) -> np.ndarray:
    """
    Preprocesses an uploaded image bytes object for the CNN model.
    1. Decodes the image from bytes.
    2. Converts BGR (OpenCV default) to RGB.
    3. Resizes to target_size (128x128).
    4. Normalizes pixel values to [0, 1].
    5. Adds batch dimension.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file or format.")
        
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to match model input shape (1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)
    
    return img
