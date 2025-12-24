import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# 1. Config & Data Loading
# ---------------------------------------------------------
# แก้ Path ให้ตรงกับเครื่องตัวเอง (ใช้ Path ที่เราเจอเมื่อกี้ว่าถูกต้อง)
dataset_path = r"C:\Users\YEDHEE\Downloads\Dataset"
IMG_SIZE = 64
image_size = (IMG_SIZE, IMG_SIZE)

def load_data(path):
    x_data = []
    y_data = []
    
    if not os.path.exists(path):
        print(f"Error: ไม่พบโฟลเดอร์ {path}")
        return np.array([]), np.array([])

    print("กำลังโหลดรูปภาพ...")
    # อ่านเฉพาะโฟลเดอร์เท่านั้น (ป้องกัน error ไฟล์ขยะ)
    # ใช้ logic ของคุณที่ช่วยกรองเฉพาะ folder จริงๆ
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    for category in classes:
        folder_path = os.path.join(path, category)
        print(f" - กำลังอ่าน Class: {category}")
        
        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                # อ่านรูปภาพและแปลงเป็นขาวดำ (Grayscale)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None: continue
                
                # Resize
                new_array = cv2.resize(img_array, image_size)
                
                # Flatten
                x_data.append(new_array.flatten())
                y_data.append(category)
            except Exception:
                pass
                
    return np.array(x_data), np.array(y_data)

X, y = load_data(dataset_path)

if len(X) > 0:
    print(f"โหลดเสร็จสิ้น! จำนวนรูปภาพทั้งหมด: {len(X)}")

    # [สำคัญมาก] Normalization: ปรับค่าสีจาก 0-255 ให้เหลือ 0-1
    # ช่วยให้ SVM ทำงานได้แม่นยำขึ้นและเรียนรู้ไวขึ้น
    X = X / 255.0

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train & Loop Kernels
    kernels = ['linear', 'poly', 'rbf']
    best_model = None
    best_acc = 0
    best_k_name = ""

    print("\nเริ่มการ Training SVM...")
    for k in kernels:
        model = SVC(kernel=k, C=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"    Kernel = {k:<6}: Accuracy = {acc:.2f}%")
        
        # เก็บโมเดลที่แม่นที่สุดจริงๆ (Dynamic Selection)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k_name = k

    print(f"\nเลือกโมเดลที่ดีที่สุด: {best_k_name} ({best_acc:.2f}%)")

    # 4. Visualization
    y_pred_best = best_model.predict(X_test)
    
    # 4.1 Sample Images
    plt.figure(figsize=(12, 4))
    num_samples = min(5, len(X_test))
    
    for i in range(num_samples):
        plt.subplot(1, 5, i+1) # แก้จาก 6 เป็น 5 ให้พอดีช่อง
        # Reshape กลับแล้วคูณ 255 เพื่อให้แสดงผลถูกต้อง (เพราะเราหารไปตอนแรก)
        # จริงๆ matplotlib show 0-1 ได้ แต่กันพลาด
        plt.imshow(X_test[i].reshape(image_size), cmap='gray')
        
        true_lb = y_test[i]
        pred_lb = y_pred_best[i]
        col = 'green' if true_lb == pred_lb else 'red'
        
        # ตัดคำให้สั้นลงหน่อยจะได้ไม่บังรูป
        plt.title(f"T:{true_lb[:5]}..\nP:{pred_lb[:5]}..", color=col, fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f"Prediction Sample (Best Kernel: {best_k_name})")
    plt.tight_layout()
    plt.show()

    # 4.2 Confusion Matrix
    print("กำลังแสดง Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    
    # พล็อตแบบถูกต้อง
    fig, ax = plt.subplots(figsize=(10, 10)) # ปรับขนาดให้ใหญ่ขึ้นหน่อยเพราะ Class ชื่อยาว
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix ({best_k_name})")
    plt.show()

else:
    print("ไม่พบข้อมูลรูปภาพ กรุณาตรวจสอบ Path")
