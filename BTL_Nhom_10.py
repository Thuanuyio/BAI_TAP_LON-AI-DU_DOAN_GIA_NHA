import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning GUI")
        self.root.geometry("800x600")
        
        self.df = None
        self.model = None  # Khởi tạo model là None để tránh lỗi
        
        # Menu
        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Exit", command=root.quit)
        menu.add_cascade(label="File", menu=file_menu)
        
        analysis_menu = tk.Menu(menu, tearoff=0)
        analysis_menu.add_command(label="Show Histogram", command=self.show_histogram)
        analysis_menu.add_command(label="Show Correlation", command=self.show_correlation)
        menu.add_cascade(label="Analysis", menu=analysis_menu)
        
        model_menu = tk.Menu(menu, tearoff=0)
        model_menu.add_command(label="Train Model", command=self.train_model)
        model_menu.add_command(label="Predict", command=self.predict_values)
        menu.add_cascade(label="Model", menu=model_menu)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            column_names = ["serisl", "date", "age", "distance", "stores", "latitude", "longitude", "price"]
            self.df = pd.read_csv(file_path, names=column_names)
            messagebox.showinfo("Success", "Data Loaded Successfully!")
        
    def show_histogram(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        fig, ax = plt.subplots()
        sns.histplot(self.df['price'], kde=True, ax=ax)
        self.display_plot(fig)
        
    def show_correlation(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        self.display_plot(fig)
        
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        self.df = self.df.iloc[:, 1:]
        num_cols = self.df.select_dtypes(include=['number']).columns
        df_norm = self.df.copy()
        df_norm[num_cols] = (self.df[num_cols] - self.df[num_cols].mean()) / self.df[num_cols].std()
        
        X = df_norm.iloc[:, :-1].values
        Y = df_norm.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
        
        self.model = Sequential([  # Gán mô hình vào self.model
            Dense(10, input_shape=(X.shape[1],), activation='relu'),
            Dense(20, activation='relu'),
            Dense(5, activation='relu'),
            Dense(1)
        ])
        self.model.compile(loss='mse', optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stopping], verbose=1)
        
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training Loss vs Validation Loss')
        self.display_plot(fig)

        messagebox.showinfo("Training Complete", "Model training completed successfully!")
        
    def predict_values(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        # Lấy danh sách các cột huấn luyện (tất cả các cột trừ cột cuối cùng, tức là "price")
        cols = self.df.columns[:-1]  # Ví dụ: ["date", "age", "distance", "stores", "latitude", "longitude"]
        num_features = self.model.input_shape[1]

        if len(cols) != num_features:
            messagebox.showerror("Error", f"Feature mismatch! Model expects {num_features} features, but got {len(cols)}.")
            return

        # Tạo mẫu dữ liệu với đúng 6 giá trị, không bao gồm giá trị "price"
        sample_values = pd.DataFrame([[2023, 5, 10, 100, 40.7128, 74.0060]], columns=cols)

        # Chuẩn hóa dữ liệu đầu vào như khi huấn luyện
        sample_values = (sample_values - self.df[cols].mean()) / self.df[cols].std()

        # Dự đoán
        prediction = self.model.predict(sample_values)
        messagebox.showinfo("Prediction Result", f"Predicted Price: {prediction[0][0]:.2f}")


        
    def display_plot(self, fig):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
