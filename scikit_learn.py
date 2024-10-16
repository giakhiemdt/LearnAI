from sklearn.linear_model import LinearRegression
import numpy as np

# Dữ liệu ví dụ
X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])

# Tạo mô hình
model = LinearRegression()
model.fit(X, y)

# Dự đoán
print(model.predict([[5]]))  # Kết quả: [6]
