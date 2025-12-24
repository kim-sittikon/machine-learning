# นำเข้า Library ที่จำเป็น
from sklearn import datasets                  # สำหรับโหลดข้อมูล Iris ที่มีมาให้แล้ว
from sklearn.model_selection import train_test_split # สำหรับแบ่งข้อมูลสอน (Train) และทดสอบ (Test)
from sklearn.svm import SVC                   # นำเข้าโมเดล SVM (Support Vector Classifier)
from sklearn.metrics import accuracy_score    # สำหรับคำนวณความแม่นยำ (Accuracy)

# 1. โหลดข้อมูล Iris Dataset
iris = datasets.load_iris()
X = iris.data    # ข้อมูลคุณลักษณะ (Features): ความยาว/ความกว้าง ของกลีบเลี้ยงและกลีบดอก
y = iris.target  # ข้อมูลคำตอบ (Labels): สายพันธุ์ของดอก Iris (0, 1, 2)

# 2. แบ่งข้อมูลเป็นชุดสอน (Train) และชุดทดสอบ (Test)
# แบ่งข้อมูล 80% สำหรับสอน และ 20% สำหรับทดสอบ (test_size=0.2)
# random_state=100 เพื่อให้ผลการสุ่มเหมือนเดิมทุกครั้งที่รัน
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# กำหนดรูปแบบของ Kernel ที่ต้องการทดสอบเปรียบเทียบ
kernels = ['linear', 'poly', 'rbf']

print("ผลการทดสอบความแม่นยำของแต่ละ Kernel:")
print("-" * 40)

# 3. วนลูปสร้างและทดสอบโมเดลทีละ Kernel
for k in kernels:
    # สร้างโมเดล SVM โดยกำหนด kernel ตามตัวแปร k และค่า C=1.0
    model = SVC(kernel=k, C=1.0, random_state=100)
    
    # 1. สั่งเทรน
    model.fit(X_train, y_train)

    # 2. ทำนายผล
    y_pred = model.predict(X_test)

    # 3. คำนวณคะแนน (ไม่ต้องคูณ 100 เพื่อให้ได้ 1.00 ตามโจทย์)
    acc = accuracy_score(y_test, y_pred)

    # 4. แสดงผลตาม Format อาจารย์
    print(f"Kernel = {k:<6}: Accuracy = {acc:.2f}")

print("-" * 40)
