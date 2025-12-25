import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings

# ปิด Warning เพื่อความสะอาดของ Output
warnings.filterwarnings('ignore')

def main():
    # ---------------------------------------------------------
    # 1. Load Face Recognition Dataset (LFW)
    # ---------------------------------------------------------
    print("Loading Face Dataset (LFW)... (อาจใช้เวลาดาวน์โหลดสักครู่ในครั้งแรก)")
    # โหลดเฉพาะคนที่มีรูปอย่างน้อย 70 รูป เพื่อให้จำแนกได้ง่ายขึ้น
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    # ข้อมูลรูปภาพ (X) และเฉลย (y)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print(f"Dataset Info: {n_samples} samples, {n_classes} classes, Image size: {h}x{w}")

    # ---------------------------------------------------------
    # 2. Split Data (Train/Test)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Normalization (ปรับค่าสีให้เป็น 0-1) ทำให้อัตโนมัติโดย Dataset อยู่แล้ว หรือทำเพิ่มได้
    # X_train = X_train / 255.0

    # ---------------------------------------------------------
    # 3. Compare Network Sizes (ตามโจทย์ 1-10 Layers, 10-1000 Nodes)
    # ---------------------------------------------------------
    print("\nComparing different network sizes... (Process might take time)")
    
    # กำหนดช่วงที่จะทดสอบ (ลดจำนวนลงเล็กน้อยเพื่อให้รันเสร็จไวขึ้นสำหรับการทดสอบ)
    # ถ้าจะเอาครบตามโจทย์เป๊ะๆ ให้แก้เป็น: layers_range = range(1, 11) และ nodes_range = [10, 50, 100, 500, 1000]
    layers_range = [1, 2, 5]      # ตัวอย่าง: ทดสอบ 1, 2, และ 5 ชั้น
    nodes_range = [10, 100, 500]  # ตัวอย่าง: ทดสอบ 10, 100, และ 500 โหนด
    
    results = []
    
    # สร้าง Model ตัวที่ดีที่สุดเก็บไว้โชว์รูป
    best_model = None
    best_acc = 0

    for layers in layers_range:
        for nodes in nodes_range:
            # สร้าง Tuple สำหรับ hidden_layer_sizes เช่น (100, 100, 100)
            hidden_layers = tuple([nodes] * layers)
            
            print(f" -> Training: {layers} Layers x {nodes} Nodes...")
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=200, random_state=42)
            mlp.fit(X_train, y_train)
            
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'Layers': layers,
                'Nodes/Layer': nodes,
                'Accuracy': acc
            })
            
            if acc > best_acc:
                best_acc = acc
                best_model = mlp

    # ---------------------------------------------------------
    # 4. Display Results Table
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot(index='Layers', columns='Nodes/Layer', values='Accuracy')
    
    print("\n" + "="*40)
    print(" Comparison Results (Accuracy) ")
    print("="*40)
    print(pivot_table)
    print("="*40)

    # ---------------------------------------------------------
    # 5. Display Sample Predictions (Visualization)
    # ---------------------------------------------------------
    if best_model is not None:
        print("\nDisplaying sample predictions form Best Model...")
        y_pred_best = best_model.predict(X_test)
        
        plt.figure(figsize=(12, 5))
        for i in range(10): # โชว์ 10 รูป
            plt.subplot(2, 5, i + 1)
            plt.imshow(X_test[i].reshape(h, w), cmap='gray')
            
            # สีเขียว=ถูก, สีแดง=ผิด
            true_name = target_names[y_test[i]].split()[-1] # เอาแค่นามสกุล
            pred_name = target_names[y_pred_best[i]].split()[-1]
            color = 'green' if y_pred_best[i] == y_test[i] else 'red'
            
            plt.title(f"T:{true_name}\nP:{pred_name}", color=color, fontsize=10)
            plt.axis('off')
            
        plt.suptitle(f"Face Recognition Results (Best Acc: {best_acc:.4f})")
        plt.tight_layout()
        plt.show()
    else:
        print("No model trained.")

if __name__ == "__main__":
    main()
