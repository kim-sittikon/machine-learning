import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("1. Configuration & Data Loading...")
base_path = r"C:\Users\YEDHEE\Downloads"
IMG_SIZE = 64
image_size = (IMG_SIZE, IMG_SIZE)
TARGET_CLASSES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def find_and_load_data(root_path, target_classes):
    x_data = []
    y_data = []
    found_classes = set()
    
    print(f"Searching for data in: {root_path} ...")
    
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.upper() in target_classes:
                print(f" -> Found folder: {dir_name}. Loading images...")
                found_classes.add(dir_name.upper())
                
                folder_path = os.path.join(root, dir_name)
                
                for img_name in os.listdir(folder_path):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                        
                    try:
                        img_path = os.path.join(folder_path, img_name)
                        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img_array is not None:
                            new_array = cv2.resize(img_array, image_size)
                            x_data.append(new_array.flatten())
                            y_data.append(dir_name.upper()) 
                    except Exception:
                        pass
    
    return np.array(x_data), np.array(y_data), found_classes

X, y, found = find_and_load_data(base_path, TARGET_CLASSES)

if len(X) > 0:
    print(f"\nLoad Complete! Found classes: {found}")
    print(f"Total images loaded: {len(X)}")

    print("Normalizing data (0-1)...")
    X = X / 255.0

    print("2. Splitting Data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kernels = ['linear', 'poly', 'rbf']
    best_model = None
    best_acc = 0
    best_k_name = ""

    print("3. Training SVM Models...")
    for k in kernels:
        print(f" -> Training Kernel: {k} ...")
        model = SVC(kernel=k, C=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"    Kernel = {k:<6}: Accuracy = {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k_name = k

    print(f"\nBest Model Selected: {best_k_name} ({best_acc:.2f}%)")

    print("4. Visualization...")
    y_pred_best = best_model.predict(X_test)
    
    print(" -> Plotting Sample Images...")
    plt.figure(figsize=(12, 4))
    num_samples = min(5, len(X_test))
    
    for i in range(num_samples):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_test[i].reshape(image_size), cmap='gray')
        
        true_lb = y_test[i]
        pred_lb = y_pred_best[i]
        col = 'green' if true_lb == pred_lb else 'red'
        
        plt.title(f"T:{true_lb[:3]}..\nP:{pred_lb[:3]}..", color=col, fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f"Blood Cells Classification (Best: {best_k_name})")
    plt.tight_layout()
    plt.show()

    print(" -> Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d', xticks_rotation='horizontal')
    plt.title(f"Confusion Matrix ({best_k_name})")
    plt.show()

else:
    print("\n[Error] No Blood Cell images found in Downloads.")
    print("Please download and extract the dataset first.")
