import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score
import os

# ตั้งค่า GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data():
    print("Loading LFW Faces Dataset...")
    try:
        # Load LFW dataset
        # min_faces_per_person=70 ensures we have enough data per class
        # resize=0.4 reduces image size for faster training
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        
        n_samples, h, w = lfw_people.images.shape
        X = lfw_people.images
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]

        print(f"Dataset Stats: {n_samples} samples, Image Size: {h}x{w}, Classes: {n_classes}")
        print(f"Classes: {target_names}")

        # Preprocessing
        # 1. Reshape to (H, W, 1) for grayscale CNN input
        X = X.reshape(n_samples, h, w, 1)
        
        # 2. Normalize (LFW is usually 0-255 or 0-1 depending on version, fetch_lfw is 0-255 usually float)
        # Check max value just in case
        if np.max(X) > 1.0:
            X = X / 255.0
            
        # 3. One-hot encoding labels
        y = tf.keras.utils.to_categorical(y, n_classes)

        # 4. Split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        return (x_train, y_train), (x_test, y_test), (h, w, 1), target_names, n_classes

    except Exception as e:
        print(f"Error loading LFW dataset: {e}")
        return None, None, None, None, None

def build_cnn_model(num_layers, num_filters, input_shape, num_classes):
    model = models.Sequential()
    
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        
        if i % 2 == 0: 
             model.add(layers.MaxPooling2D((2, 2), padding='same'))
             # Add dropout to prevent overfitting on small dataset
             model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test), input_shape, target_names, n_classes = load_data()
    
    if x_train is None:
        return

    # 2. Setup Experiment
    # ใช้ Range ตาม Lab 1 เพื่อเปรียบเทียบ
    layers_to_test = [1, 2, 3]       
    filters_to_test = [32, 64]      
    
    results = []
    
    print(f"\nStarting Experiment on Face Recognition...")
    
    best_model = None
    best_acc = 0
    
    for l in layers_to_test:
        for f in filters_to_test:
            print(f"\n--- Training CNN: {l} Layers, {f} Filters ---")
            
            try:
                model = build_cnn_model(l, f, input_shape, n_classes)
                
                # Train (Face Recognition on LFW is harder, need more epochs usually)
                history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
                
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
    print(" Experiment Results: CNN Face Recognition")
    print("="*50)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        pivot_table = df_results.pivot(index='Conv Layers', columns='Filters/Nodes', values='Test Accuracy')
        print(pivot_table)
    print("="*50)
    
    # 4. Display Predictions (Best Model)
    if best_model:
        print("\nDisplaying Sample Predictions from Best Model...")
        predictions = best_model.predict(x_test[:5])
        predicted_ids = np.argmax(predictions, axis=1)
        true_ids = np.argmax(y_test[:5], axis=1)
        
        plt.figure(figsize=(15, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(x_test[i].reshape(input_shape[0], input_shape[1]), cmap='gray')
            
            pred_name = target_names[predicted_ids[i]].split()[-1] # Last name
            true_name = target_names[true_ids[i]].split()[-1]
            
            color = 'green' if predicted_ids[i] == true_ids[i] else 'red'
            plt.title(f"P:{pred_name}\nT:{true_name}", color=color)
            plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
