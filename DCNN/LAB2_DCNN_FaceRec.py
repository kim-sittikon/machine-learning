import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(target_size=(100, 100)):
    print(f"\n[INFO] Loading LFW dataset and resizing to {target_size}...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=None, data_home=r"C:\Users\YEDHEE\Desktop\machine learning\Set Load\LFW_Face_Dataset") # Load original size
    
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print(f"Total samples: {n_samples}")
    print(f"Image size (original): {h}x{w}")
    print(f"Classes: {n_classes} {target_names}")

    # Resize and convert to RGB
    X_resized = []
    for img in X:
        # LFW images are float32 in [0, 255] or [0, 1] usually. Scikit load returns [0, 255] for raw images usually but let's check max.
        # Actually fetch_lfw_people return float. Let's normalize to 0-255 for cv2 resize then back.
        # But safest is:
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img_res = cv2.resize(img_norm, target_size)
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_GRAY2RGB) # VGG/ResNet expect 3 channels
        X_resized.append(img_rgb)
    
    X_resized = np.array(X_resized)
    
    # Normalize to [0, 1]
    X_resized = X_resized.astype('float32') / 255.0
    
    # One-hot encoding
    y_cat = to_categorical(y, num_classes=n_classes)
    
    return X_resized, y_cat, y, target_names, n_classes

def build_model(model_name, input_shape, n_classes):
    print(f"[INFO] Building {model_name}...")
    
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Transfer Learning: Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    image_sizes = [(50, 50), (100, 100)]
    models_to_train = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']
    
    results = []

    for size in image_sizes:
        print(f"\n{'='*40}")
        print(f"Processing Image Size: {size}")
        print(f"{'='*40}")
        
        X, y_cat, y_raw, target_names, n_classes = load_and_preprocess_data(target_size=size)
        input_shape = (size[0], size[1], 3)
        
        # Split: Train (70%), Val (15%), Test (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42, stratify=y_raw)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))
        
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
        
        for model_name in models_to_train:
            K.clear_session() # Clear memory
            
            try:
                model = build_model(model_name, input_shape, n_classes)
                
                print(f"Training {model_name} on {size}...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10, # Keep epochs low for lab demo
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate
                print(f"Evaluating {model_name}...")
                loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)
                loss_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
                loss_val, acc_val = model.evaluate(X_val, y_val, verbose=0)
                
                # Predictions for metrics
                y_pred_prob = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_true = np.argmax(y_test, axis=1)
                
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Save Model
                save_dir = os.path.join(os.path.dirname(__file__), 'models')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"model_{model_name}_{size[0]}x{size[1]}.h5")
                model.save(save_path)
                print(f"Model saved to {save_path}")
                
                results.append({
                    'Image Size': f"{size[0]}x{size[1]}",
                    'Model': model_name,
                    'Train Acc': acc_train,
                    'Val Acc': acc_val,
                    'Test Acc': acc_test,
                    'Precision': prec,
                    'Recall': rec
                })
                
                # Sample Predictions (Only for last model of last size or first one to save time? Let's do for 100x100 VGG)
                if size == (100, 100) and model_name == 'VGG16':
                    print("\n[INFO] Displaying sample predictions for VGG16 (100x100)...")
                    plt.figure(figsize=(10, 5))
                    for i in range(5):
                        plt.subplot(1, 5, i+1)
                        plt.imshow(X_test[i])
                        plt.title(f"True: {target_names[y_true[i]].split()[-1]}\nPred: {target_names[y_pred[i]].split()[-1]}")
                        plt.axis('off')
                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                print(f"Failed to train {model_name} on {size}: {e}")

    # Output Comparison Table
    print(f"\n{'='*20} PERFORMANCE COMPARISON LAB2 {'='*20}")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Analysis
    print(f"\n{'='*20} ANALYSIS: Face Recognition vs Classification {'='*20}")
    print("""
    Explanation:
    1. Face Classification (Face Identification):
       - As implemented in this Lab, we treat each person as a 'Class'.
       - The model learns to classify an input image into one of N predefined identities.
       - It uses Softmax output where dimensionality = Number of People.
       - Limitation: Requires retraining if a new person is added.
    
    2. Face Recognition (Verification/Embedding):
       - Often learnt via Metric Learning (Siamese Networks, Triplet Loss).
       - The model outputs a feature vector (embedding).
       - To recognize a face, we compare the distance between embeddings.
       - Advantage: Can handle new people without retraining (Open Set Recognition).
       
    In this LAB, we used the 'Classification' approach (Closed Set) using Transfer Learning of DCNNs.
    """)

if __name__ == "__main__":
    main()
