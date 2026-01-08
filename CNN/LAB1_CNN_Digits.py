import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ตั้งค่า GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data():
    print("Loading MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocessing: Reshape (28, 28, 1) & Normalize (0-1)
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # One-hot Encoding labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Split Train/Val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_cnn_model(num_layers, num_filters, input_shape=(28, 28, 1)):
    model = models.Sequential()
    
    # 1. Add Convolutional Layers dynamically
    for i in range(num_layers):
        if i == 0:
            # Layer แรกต้องระบุ input_shape
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        
        # เพิ่ม MaxPool ทุกๆ 2 conv layers เพื่อลดขนาดภาพ (ป้องกันภาพเล็กเกินไปจน error)
        # หรือถ้า layer เยอะมากอาจจะใส่แค่บางจุด อันนี้ใส่แบบง่ายๆ คือถ้าเป็นเลขคี่ให้ใส่ Pool
        if i % 2 == 0: 
             model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # 2. Flatten & Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # Output 10 digits
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    
    # 2. Setup Experiment
    # โจทย์บอกให้ลอง 1-10 layers และ 10-1000 nodes
    # เพื่อความรวดเร็วในการทดสอบเบื้องต้น เราจะใช้ subset ของค่าเหล่านี้
    # หากต้องการทดสอบเต็มรูปให้แก้เป็น range(1, 11) และ [10, 100, 1000]
    layers_to_test = [1, 2, 3]       
    filters_to_test = [32, 64]      
    
    results = []
    
    print(f"\nStarting Experiment...")
    print(f"Layers to test: {layers_to_test}")
    print(f"Filters (Nodes) to test: {filters_to_test}")
    
    best_model = None
    best_acc = 0
    
    for l in layers_to_test:
        for f in filters_to_test:
            print(f"\n--- Training CNN: {l} Layers, {f} Filters ---")
            
            try:
                model = build_cnn_model(num_layers=l, num_filters=f)
                
                # Train (จำนวน Epochs น้อยๆ เพื่อทดสอบเร็วๆ)
                history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_val, y_val), verbose=1)
                
                # Evaluate
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                
                print(f"Result: Accuracy = {test_acc:.4f}")
                results.append({
                    'Conv Layers': l,
                    'Filters/Nodes': f,
                    'Test Accuracy': test_acc
                })
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model = model
                    
            except Exception as e:
                print(f"Error training {l} layers, {f} filters: {e}")

    # 3. Output Table
    print("\n" + "="*50)
    print(" Experiment Results: CNN Architecture Comparison")
    print("="*50)
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        pivot_table = df_results.pivot(index='Conv Layers', columns='Filters/Nodes', values='Test Accuracy')
        print(pivot_table)
    else:
        print("No results to display.")
    print("="*50)
    
    # 4. Display Predictions (Best Model)
    if best_model:
        print("\nDisplaying Sample Predictions from Best Model...")
        predictions = best_model.predict(x_test[:10])
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test[:10], axis=1)
        
        plt.figure(figsize=(15, 3))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
            plt.title(f"P:{predicted_labels[i]}\nT:{true_labels[i]}", color=color)
            plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
