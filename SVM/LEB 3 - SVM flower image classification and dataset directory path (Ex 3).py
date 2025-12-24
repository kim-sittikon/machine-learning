import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. Config & Data Loading
# ---------------------------------------------------------
# Path ไปยังโฟลเดอร์ Dataset (ต้องแตกไฟล์แล้วจะได้โฟลเดอร์ย่อยๆ ตามชื่อดอกไม้)
# Link Dataset: https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision
dataset_path = r"C:\Users\YEDHEE\Downloads" 

# กำหนดขนาดรูปภาพที่จะ Resize (ลดขนาดเพื่อให้ Training ไวขึ้น)
IMG_SIZE = 64  
image_size = (IMG_SIZE, IMG_SIZE)

X = []
y = []

# print("Loading images...")

# ตรวจสอบว่ามี Folder Dataset อยู่จริงไหม
if not os.path.exists(dataset_path):
    print(f"Error: Path not found: {dataset_path}")
    print("Please download dataset from Kaggle and set the correct path.")
else:
    # อ่านไฟล์จาก Folder ย่อยแต่ละ Class
    # ระบุชื่อ Class ตรงๆ เพื่อป้องกันไปอ่าน Folder อื่นใน Downloads
    # ตรวจสอบว่าโฟลเดอร์เหล่านี้มีอยู่จริงใน Downloads ของคุณหรือไม่
    classes = ['iris-setosa', 'iris-versicolor', 'iris-virginica'] 
    for category in classes:
        path = os.path.join(dataset_path, category)
        if not os.path.isdir(path):
            continue
            
        # print(f"Loading class: {category}")
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # อ่านรูปภาพและแปลงเป็นขาวดำ (Grayscale)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue
                    
                # Resize รูปภาพ
                new_array = cv2.resize(img_array, image_size)
                
                # Flatten รูปภาพ (เปลี่ยนจาก 2D array เป็น 1D array) เพื่อใส่ใน SVM
                X.append(new_array.flatten())
                y.append(category)
            except Exception as e:
                pass

    X = np.array(X)
    y = np.array(y)
    
    # แปลง Label จากตัวหนังสือเป็นตัวเลข (ถ้าจำเป็น) หรือใช้ string เลยก็ได้สำหรับ SVM sklearn
    # แต่ในที่นี้จะเก็บไว้เป็น string เพื่อใช้แสดงผลตอน plot

    # print(f"Total images loaded: {len(X)}")

    if len(X) > 0:
        # ---------------------------------------------------------
        # 2. Split Train/Test
        # ---------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # ---------------------------------------------------------
        # 3. Train SVM Model
        # ---------------------------------------------------------
        # print("Training SVM model...")
        model = SVC(kernel='linear', C=1.0, random_state=42) # ใช้ Linear Kernel หรือลองเปลี่ยนเป็น 'rbf'
        model.fit(X_train, y_train)

        # ---------------------------------------------------------
        # 4. Prediction & Evaluation
        # ---------------------------------------------------------
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"Model Accuracy on Test Set: {acc:.2f}%")

        # ---------------------------------------------------------
        # 5. Visualization (ตามโจทย์)
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 3))
        
        # สุ่มเลือกรูปมาแสดง หรือจะเอา 5 รูปแรกก็ได้
        # ในที่นี้เอา 5 รูปแรกของ Test set ตาม Loop
        num_samples = min(5, len(X_test))
        
        for i in range(num_samples):
            idx = i
            
            plt.subplot(1, 5, i+1)
            
            # Reshape กลับเป็น 2D เพื่อแสดงผลรูปภาพ
            plt.imshow(X_test[idx].reshape(image_size), cmap='gray')
            
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            
            # กำหนดสีตัวหนังสือ: ถูก=เขียว, ผิด=แดง
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("No images found. Please check dataset path.")
