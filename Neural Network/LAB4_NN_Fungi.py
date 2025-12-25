import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import warnings

# ปิด Warning เพื่อความสะอาดของ Output
warnings.filterwarnings('ignore')

def load_images_from_folder(folder, image_size=(32, 32)):
    images = []
    labels = []
    
    if not os.path.exists(folder):
        print(f"Error: Folder not found: {folder}")
        return np.array(images), np.array(labels)
        
    print(f"Scanning folder: {folder}")
    
    # วนลูปตามโฟลเดอร์ของแต่ละคลาส (class_name)
    for filename in os.listdir(folder):
        class_path = os.path.join(folder, filename)
        
        # ถ้าเป็นโฟลเดอร์ (คือ Class)
        if os.path.isdir(class_path):
            class_name = filename
            # อ่านรูปภาพใน Class นั้นๆ
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        # Resize รูปให้เล็กลงเพื่อลดจำนวน Features (เช่น 32x32 = 1024 pixels)
                        img = img.resize(image_size).convert('RGB')
                        # แปลงเป็น numpy array และ Normalize (0-1)
                        img_array = np.array(img).flatten() / 255.0
                        
                        images.append(img_array)
                        labels.append(class_name)
                except Exception as e:
                    pass
        # กรณี dataset ไม่มี subfolder (รูปกองรวมกัน) - ข้ามไปก่อน หรือเขียน logic เพิ่ม
        
    return np.array(images), np.array(labels)

def main():
    print("="*50)
    print(" LAB 4: NN on Microscopic Fungi Classification")
    print("="*50)

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    # กำหนดตำแหน่ง dataset
    train_dir = "../dataset/fungi/train"
    test_dir = "../dataset/fungi/test"
    
    # Fallback paths
    if not os.path.exists(train_dir): train_dir = "dataset/fungi/train"
    if not os.path.exists(test_dir): test_dir = "dataset/fungi/test"
    
    print("\nLoading Training Data...")
    X_train, y_train_labels = load_images_from_folder(train_dir, image_size=(32, 32))
    
    print("\nLoading Testing Data...")
    X_test, y_test_labels = load_images_from_folder(test_dir, image_size=(32, 32))

    if len(X_train) == 0:
        print("Error: No training data found. Please check dataset path.")
        return

    print(f"\nData Loaded:")
    print(f"Train: {X_train.shape} samples")
    print(f"Test:  {X_test.shape} samples")
    
    # Encode Labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_labels)
    y_test = le.transform(y_test_labels)
    
    classes = le.classes_
    print(f"Classes: {classes}")

    # ---------------------------------------------------------
    # 2. Experiment 1: 5 Layers (100 Nodes each)
    # ---------------------------------------------------------
    print("\n" + "-"*50)
    print(" Experiment 1: 5 Layers x 100 Nodes")
    print("-" * 50)
    
    mlp_base = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=200, random_state=42)
    mlp_base.fit(X_train, y_train)
    
    acc_base = mlp_base.score(X_test, y_test)
    print(f"Base Model Accuracy: {acc_base:.4f}")

    # ---------------------------------------------------------
    # 3. Experiment 2: Compare Network Sizes
    # ---------------------------------------------------------
    print("\n" + "-"*50)
    print(" Experiment 2: Compare Network Sizes (Layers & Nodes)")
    print("-" * 50)
    
    # เพื่อความรวดเร็วในการ Lab เราจะทดสอบสเกลเล็กๆ (ถ้าจะเอาครบ 1-10 Layers ต้องใช้เวลานานมาก)
    layers_test = [1, 3, 5]
    nodes_test = [10, 50, 100]
    
    results_size = []
    
    for l in layers_test:
        for n in nodes_test:
            hidden_layers = tuple([n] * l)
            print(f"Training: {l} Layers, {n} Nodes...")
            
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=100, random_state=42)
            mlp.fit(X_train, y_train)
            acc = mlp.score(X_test, y_test)
            
            results_size.append({'Layers': l, 'Nodes': n, 'Accuracy': acc})
            
    df_size = pd.DataFrame(results_size)
    pivot_size = df_size.pivot(index='Layers', columns='Nodes', values='Accuracy')
    print("\nNetwork Size Comparison (Accuracy):")
    print(pivot_size)

    # ---------------------------------------------------------
    # 4. Experiment 3: Compare Learning Rates
    # ---------------------------------------------------------
    print("\n" + "-"*50)
    print(" Experiment 3: Compare Learning Rates (10^-2 to 10^-5)")
    print("-" * 50)
    
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    results_lr = []
    
    best_model = None
    best_acc = 0

    # ใช้โครงสร้างเดิม (5 Layers, 100 Nodes) เพื่อทดสอบ LR
    fixed_structure = (100, 100, 100, 100, 100)

    for lr in learning_rates:
        print(f"Training LR: {lr}...")
        mlp = MLPClassifier(hidden_layer_sizes=fixed_structure, learning_rate_init=lr, max_iter=200, random_state=42)
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        
        results_lr.append({
            'Learning Rate': lr,
            'LR Scientific': f"10^{int(np.log10(lr))}",
            'Accuracy': acc
        })
        
        if acc >= best_acc:
            best_acc = acc
            best_model = mlp

    df_lr = pd.DataFrame(results_lr)[['LR Scientific', 'Learning Rate', 'Accuracy']]
    print("\nLearning Rate Comparison:")
    print(df_lr)

    # ---------------------------------------------------------
    # 5. Sample Predictions
    # ---------------------------------------------------------
    print("\n" + "-"*50)
    print(" Sample Predictions (Best Model)")
    print("-" * 50)
    
    # สุ่มรูปมาแสดง 10 รูป
    indices = np.random.choice(len(X_test), 10, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        image = X_test[idx].reshape(32, 32, 3) # Reshape กลับเป็นรูปภาพ (32x32x3)
        true_label = le.inverse_transform([y_test[idx]])[0]
        pred_label = le.inverse_transform([best_model.predict([X_test[idx]])])[0]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.subplot(2, 5, i+1)
        plt.imshow(image)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
