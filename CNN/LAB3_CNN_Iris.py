import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import os

# ตั้งค่า GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(file_path):
    print(f"Loading Dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # แยก Features และ Labels
        # สมมติว่า Column สุดท้ายคือ Label (Species)
        X = df.iloc[:, 1:-1].values # ตัด Id (ถ้ามี) และ Label
        y = df.iloc[:, -1].values
        
        # ถ้ารูปแบบไฟล์ Iris มาตรฐานมักจะมี Id เป็น column แรก หรือไม่มีเลย
        # ลองเช็คว่า column แรกเป็นตัวเลขเรียงกันไหม ถ้าใช่ให้ตัดออก
        if 'Id' in df.columns or 'id' in df.columns:
             X = df.iloc[:, 1:-1].values
        else:
             X = df.iloc[:, :-1].values
             
        # Encode Label
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(np.unique(y))
        
        # Scale Data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape for Conv1D: (Samples, Features, 1)
        # เรามี 4 Features (SepalLength, SepalWidth, PetalLength, PetalWidth)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # One-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes)
        
        # Split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Data Shape: {X.shape}, Classes: {num_classes}")
        return (x_train, y_train), (x_test, y_test), num_classes, le
        
    except Exception as e:
        print(f"Error loading Iris dataset: {e}")
        return None, None, None, None

def build_cnn1d_model(num_layers, num_filters, input_shape, num_classes, learning_rate=0.001):
    model = models.Sequential()
    
    # 1. Add Conv1D Layers
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv1D(num_filters, kernel_size=2, activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv1D(num_filters, kernel_size=2, activation='relu', padding='same'))
        
        # MaxPool 1D (Optional, Iris features are few so maybe skip or use size 1/2)
        # เนื่องจาก Feature น้อย (4) ถ้า pool มากไปจะเหลือ 0 ทำให้ error
        # ใส่ Pool แค่ layer แรกๆ หรือไม่ใส่เลยสำหรับข้อมูลน้อยๆ
        if i == 0 and input_shape[0] > 2:
             model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Path to Iris.csv
    file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Iris.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        # ลองหาแบบ relative path ปกติ
        file_path = 'dataset/Iris.csv'
        if not os.path.exists(file_path):
             print(f"File strictly not found at {file_path}. Please check path.")
             return

    # 1. Load Data
    (x_train, y_train), (x_test, y_test), num_classes, le = load_data(file_path)
    
    if x_train is None: return

    input_shape = (x_train.shape[1], 1)

    # --- Experiment 1: Learning Rates ---
    print("\n" + "="*50)
    print(" Experiment 1: Different Learning Rates")
    print(" (Fixed Architecture: 2 Layers, 32 Filters)")
    print("="*50)
    
    lrs_to_test = [0.01, 0.001, 0.0001, 0.00001] # 10^-2 to 10^-5
    results_lr = []
    
    for lr in lrs_to_test:
        print(f"Testing Learning Rate: {lr}")
        model = build_cnn1d_model(num_layers=2, num_filters=32, input_shape=input_shape, num_classes=num_classes, learning_rate=lr)
        model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0) # Silent training
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f" -> Accuracy: {acc:.4f}")
        results_lr.append({'Learning Rate': lr, 'Accuracy': acc})

    df_lr = pd.DataFrame(results_lr)
    print("\nResults Table (Learning Rate):")
    print(df_lr)

    # --- Experiment 2: Network Sizes ---
    print("\n" + "="*50)
    print(" Experiment 2: Network Sizes (Layers & Nodes)")
    print(" (Fixed Learning Rate: 0.001)")
    print("="*50)
    
    layers_to_test = [1, 2, 3] # สามารถเพิ่มถึง 10 ได้
    filters_to_test = [16, 32, 64, 128] # สามารถเพิ่มถึง 1000 ได้
    results_size = []
    
    for l in layers_to_test:
        for f in filters_to_test:
            print(f"Testing: {l} Layers, {f} Filters...")
            try:
                model = build_cnn1d_model(num_layers=l, num_filters=f, input_shape=input_shape, num_classes=num_classes)
                model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=0)
                loss, acc = model.evaluate(x_test, y_test, verbose=0)
                results_size.append({'Layers': l, 'Filters': f, 'Accuracy': acc})
            except Exception as e:
                print(f"Error: {e}")

    df_size = pd.DataFrame(results_size)
    pivot_size = df_size.pivot(index='Layers', columns='Filters', values='Accuracy')
    print("\nResults Table (Network Sizes):")
    print(pivot_size)
    print("="*50)

if __name__ == "__main__":
    main()
