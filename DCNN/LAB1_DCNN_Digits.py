import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import os


# ตั้งค่า GPU (ถ้ามี)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_and_preprocess_data(image_size=(100, 100), subset_size=None):
    """
    โหลดข้อมูล MNIST และเตรียมข้อมูล
    - image_size: ขนาดภาพที่ต้องการ (default 100x100)
    - subset_size: จำนวนข้อมูลที่จะใช้ (None = ใช้ทั้งหมด) เพื่อความรวดเร็วในการทดสอบ
    """
    print(f"กำลังโหลดข้อมูล MNIST และเตรียมภาพขนาด {image_size}...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # ถ้ากำหนด subset_size ให้ลดจำนวนข้อมูลลง (สำหรับการทดสอบที่รวดเร็ว)
    if subset_size:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_test = x_test[:subset_size//5]
        y_test = y_test[:subset_size//5]

    # รวมข้อมูลเพื่อแบ่งใหม่ (Train/Val/Test)
    X = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    # Preprocessing images
    # เนื่องจาก DCNN ส่วนใหญ่ต้องการ 3 channels (RGB) เราต้องแปลง Grayscale -> RGB
    # และ Resize ภาพ
    
    # การ Resize จะทำใน tf.data.Dataset เพื่อประหยัด Memory
    # แต่ที่นี่เราจะแบ่งข้อมูลก่อน
    
    # Split Dataset: Train 70%, Val 15%, Test 15%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # One-hot encoding for metrics calculation during training
    y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 10)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    print(f"Data Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat), (X_test, y_test, y_test_cat)

def preprocess_image(image, label, target_size):
    """ฟังก์ชันสำหรับแปลงภาพใน tf.data.Dataset"""
    image = tf.expand_dims(image, -1) # Add channel dim
    image = tf.image.grayscale_to_rgb(image) # Convert to 3 channels
    image = tf.image.resize(image, target_size) # Resize
    image = image / 255.0 # Normalize
    return image, label

def create_dataset(X, y, batch_size=32, target_size=(100, 100)):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y, target_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(model_name, input_shape, num_classes=10):
    """สร้างโมเดลตามชื่อที่ระบุ"""
    base_model = None
    
    if model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=input_shape)
    elif model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    elif model_name == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Add custom head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Configuration
    IMG_SIZE = (100, 100) # หรือ (200, 200) ตามโจทย์
    BATCH_SIZE = 32
    EPOCHS = 5 # จำนวนรอบการเทรน (ปรับเพิ่มได้)
    SUBSET_SIZE = 2000 # ใช้ข้อมูลแค่ 2000 ตัวอย่างเพื่อการทดสอบที่รวดเร็ว (ถ้าต้องการ Full ให้ตั้งเป็น None)
    
    # 1. Load Data
    (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat), (X_test, y_test, y_test_cat) = load_and_preprocess_data(IMG_SIZE, subset_size=SUBSET_SIZE)

    # 2. Create tf.data.Datasets
    train_ds = create_dataset(X_train, y_train_cat, BATCH_SIZE, IMG_SIZE)
    val_ds = create_dataset(X_val, y_val_cat, BATCH_SIZE, IMG_SIZE)
    # Test dataset for strict evaluation
    test_ds = create_dataset(X_test, y_test_cat, BATCH_SIZE, IMG_SIZE)

    models_to_train = ['MobileNetV2', 'VGG16', 'ResNet50', 'DenseNet121'] # เรียงจากเบาไปหนัก
    results = []

    if not os.path.exists('models'):
        os.makedirs('models')

    for model_name in models_to_train:
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        input_shape = IMG_SIZE + (3,)
        
        try:
            model = build_model(model_name, input_shape)
            
            # Train
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
            
            # Save Model
            model.save(f"models/{model_name}_mnist.h5")
            print(f"Saved {model_name} to models/{model_name}_mnist.h5")
            
            # Evaluate on Test Set
            print(f"Evaluating {model_name} on Test Set...")
            # Predict
            # ต้อง Loop dataset เพื่อเอา gt labels หรือใช้ y_test_cat
            y_pred_probs = model.predict(test_ds)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test_cat, axis=1) # หรือ y_test (เช็คขนาดให้ตรงกัน)
               
            # Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            print(f"Results for {model_name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec
            })
            
            # Sample Predictions
            print(f"Sample Predictions ({model_name}):")
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            ids = np.random.choice(len(X_test), 5, replace=False)
            for i, idx in enumerate(ids):
                img = X_test[idx] # Original image (28x28)
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
                axes[i].axis('off')
            plt.suptitle(f'Sample Predictions: {model_name}')
            plt.tight_layout()
            plt.show() # อาจจะไม่แสดงผลถ้าปิดหน้าต่างกราฟ เรารับทราบ
            plt.close()

        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # 3. Compare Performance Table
    print("\n" + "="*40)
    print("      Model Performance Comparison      ")
    print("="*40)
    df_results = pd.DataFrame(results)
    print(df_results)
    print("="*40)

    # 4. Analysis (Thai)
    print("\n" + "#"*50)
    print("             บทวิเคราะห์และการสรุปผล             ")
    print("#"*50)
    print("""
หัวข้อ: ความแตกต่างระหว่าง Digit Recognition และ Digit Classification

1. **Digit Recognition (การรู้จำตัวเลข)**:
   - เป็นกระบวนการที่กว้างกว่า ซึ่งรวมถึงการระบุตำแหน่งของตัวเลขในภาพ (Localization) และการระบุว่าเป็นตัวเลขอะไร
   - ขอบเขตงานมักจะซับซ้อนกว่า เช่น การอ่านตัวเลขลายมือเขียนจากเอกสารทั้งใบ (OCR) ซึ่งต้องมีการ Segmentation เพื่อแยกตัวอักษรก่อน
   - เป้าหมายคือการทำให้คอมพิวเตอร์ "อ่าน" และ "เข้าใจ" ภาพที่มีตัวเลขประกอบอยู่

2. **Digit Classification (การจำแนกประเภทตัวเลข)**:
   - เป็นสับเซต (Subset) ของการรู้จำ
   - โจทย์เจาะจงที่: "ภาพ input นี้ (ซึ่งรู้แล้วว่าเป็นภาพตัวเลข 1 ตัว) คือเลขโดดอะไร (0-9)?"
   - ใน Lab นี้ สิ่งที่เราทำคือ Classification เพราะ input ของเราเป็นภาพ MNIST ที่ถูก Crop และ Center มาแล้ว หน้าที่ของโมเดลคือแค่จัดหมวดหมู่ (Classify) ว่าเป็น Class ไหน
   
สรุป: DCNN เหมาะกับงาน Image Classification มาก เนื่องจากความสามารถในการสกัด Feature (Feature Extraction) ผ่าน Convolutional Layers ทำให้สามารถเรียนรู้ลักษณะสำคัญของตัวเลข (เส้นโค้ง, เส้นตรง, มุม) ได้โดยอัตโนมัติ ไม่ว่าภาพจะมีการเลื่อนหรือเปลี่ยนขนาดเล็กน้อย
    """)

if __name__ == "__main__":
    main()
