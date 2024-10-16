import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

# Đường dẫn tới dataset
data_dir = 'C:/Users/khiem/Downloads/archive/leapGestRecog/00'

# Các lớp cử chỉ
classes = ['01_palm', '02_l', '03_fist', '04_fist_moved', "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
img_size = (640, 240)  # Kích thước ảnh mới

# Chuẩn bị dữ liệu
data = []
labels = []

for label, gesture in enumerate(classes):
    folder_path = os.path.join(data_dir, gesture)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)  # Resize ảnh về kích thước 640x240
        data.append(img)
        labels.append(label)

# Chuyển dữ liệu thành mảng numpy và chuẩn hóa
data = np.array(data).reshape(-1, img_size[1], img_size[0], 1) / 255.0
labels = np.array(labels)

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành one-hot encoding
y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(y_test, len(classes))

# Xây dựng mô hình CNN
model = Sequential()

# Lớp tích chập đầu tiên
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 640, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp tích chập thứ hai
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp tích chập thứ ba
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten và kết nối với các lớp Fully Connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Hiển thị cấu trúc mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Dự đoán với ảnh mới
def predict_gesture(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size).reshape(-1, img_size[1], img_size[0], 1) / 255.0
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

# Ví dụ dự đoán
img_path = 'sample.png'  # Đường dẫn đến ảnh mẫu
gesture = predict_gesture(img_path)
print(f'Predicted Gesture: {gesture}')

# Lưu mô hình
model.save('hand_gesture_model.h5')

# Tải lại mô hình
from tensorflow.keras.models import load_model # type: ignore
model = load_model('hand_gesture_model.h5')
