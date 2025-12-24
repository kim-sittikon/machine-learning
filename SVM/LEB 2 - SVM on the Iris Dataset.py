import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data from CSV
# ต้องระวัง path ไฟล์ให้ถูกต้องนะครับ
df = pd.read_csv(r"D:\code\project_\iris.csv") 

# ลองปริ้นหัวตารางออกมาดูว่าข้อมูลหน้าตาเป็นไง
print(df.head())

# Features and target
# ------------------ ส่วนที่ต้องเติม 1 ------------------
# X คือ data ทั้งหมดยกเว้นคอลัมน์สุดท้าย (ใช้ iloc[:, :-1])
X = df.iloc[:, :-1].values   
# -----------------------------------------------------

y = df.iloc[:, -1].values    # last column is label

# 2. Split train/test
# รอบนี้ใช้ random_state=42 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Compare kernels
kernels = ['linear', 'poly', 'rbf']

for k in kernels:
    # ------------------ ส่วนที่ต้องเติม 2 ------------------
    
    # สร้างโมเดล SVM ตาม kernel รอบนั้นๆ
    model = SVC(kernel=k, C=1.0, random_state=42)

    # สั่งให้โมเดลเรียนรู้จากข้อมูลชุด Train
    model.fit(X_train, y_train)

    # ให้โมเดลลองทำนายผลจากชุด Test
    y_pred = model.predict(X_test)

    # คิดคะแนนความแม่นยำ (คูณ 100 เพื่อให้เป็น %)
    acc = accuracy_score(y_test, y_pred) * 100

    # ปริ้นผลลัพธ์ตาม Format ที่โจทย์ต้องการ (มี %)
    print(f"Kernel = {k:<6}: Accuracy = {acc:.2f}%")

    # -----------------------------------------------------