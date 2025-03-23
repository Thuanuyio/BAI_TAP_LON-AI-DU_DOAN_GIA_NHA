import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Định nghĩa danh sách tên cột
column_names = ["serial", "date", "age", "distance", "stores", "latitude", "longitude", "price"]

# Import dữ liệu
df = pd.read_csv('Data_Set.csv', names=column_names)
df.head()

# Hiển thị phân bố giá
sns.histplot(df['price'], kde=True)
plt.show()

# Hiển thị ma trận tương quan
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Kiểm tra dữ liệu bị thiếu
print("Missing values:\n", df.isna().sum())

# Chuẩn bị dữ liệu
df = df.iloc[:, 1:]  # Giữ lại các cột từ thứ 2 trở đi (nếu cần)

# Chuẩn hóa dữ liệu (chỉ áp dụng với cột số)
df_norm = df.copy()
num_cols = df.select_dtypes(include=['number']).columns
df_norm[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
df_norm.head()

y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

# Tách dữ liệu thành X (đặc trưng) và Y (nhãn)
X = df_norm.iloc[:, :-1].values
Y = df_norm.iloc[:, -1].values

print('X shape:', X.shape)
print('Y shape:', Y.shape)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, shuffle=True, random_state=0)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Hàm tạo model
def get_model():
    model = Sequential([
        Dense(10, input_shape=(X.shape[1],), activation='relu'),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# **Dự đoán trước khi huấn luyện**
model_untrained = get_model()  # Tạo model mới chưa huấn luyện
preds_on_untrained = model_untrained.predict(X_test)  # Dự đoán trước khi huấn luyện

# Chuyển đổi về giá trị gốc
price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]

# **Huấn luyện mô hình chính**
model = get_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

# Vẽ đồ thị mất mát
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss vs Validation Loss')
plt.show()

# Dự đoán sau khi huấn luyện
preds_on_trained = model.predict(X_test)

# Chuyển đổi dự đoán về giá trị gốc
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

# So sánh kết quả trước và sau huấn luyện
plt.figure(figsize=(10, 5))
plt.plot(price_y_test, label="Actual Price", marker="o")
plt.plot(price_on_trained, label="Predicted Price", marker="x")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.legend()
plt.title("Actual vs Predicted Price")
plt.show()

# Vẽ biểu đồ so sánh dự đoán trước và sau khi huấn luyện
plt.figure(figsize=(8, 6))

# Vẽ điểm dữ liệu của mô hình chưa huấn luyện (màu đỏ)
plt.scatter(price_on_untrained, price_y_test, color='red', label='Untrained Model', alpha=0.6)

# Vẽ điểm dữ liệu của mô hình đã huấn luyện (màu xanh lá)
plt.scatter(price_on_trained, price_y_test, color='green', label='Trained Model', alpha=0.6)

# Vẽ đường y = x (đường kỳ vọng nếu dự đoán hoàn hảo)
plt.plot([min(price_y_test), max(price_y_test)], [min(price_y_test), max(price_y_test)], 'b--', label='Ideal Prediction')

# Thêm nhãn và tiêu đề
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.legend()  # Hiển thị chú thích
plt.title("Comparison of Model Predictions")

plt.show()
