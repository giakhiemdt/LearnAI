from tensorflow.keras.models import load_model # type: ignore
import os
import cv2
import numpy as np

model = load_model('hand_gesture_model.h5')
img_size = (640, 240)  # Kích thước ảnh mới
classes = ['01_palm', '02_l', '03_fist', '04_fist_moved', "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]

def predict_gesture(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size).reshape(-1, img_size[1], img_size[0], 1) / 255.0
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

img_path = 'Test1.png'  # Đường dẫn đến ảnh mẫu
gesture = predict_gesture(img_path)
print(f'Predicted Gesture: {gesture}')