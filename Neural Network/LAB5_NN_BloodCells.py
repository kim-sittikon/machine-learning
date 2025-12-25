import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

def load_data(root_folder, image_size=(64, 64), max_samples_per_class=None):
    images = []
    labels = []
    
    if not os.path.exists(root_folder):
        print(f"Error: Folder not found: {root_folder}")
        return np.array(images), np.array(labels)
        
    print(f"Scanning folder: {root_folder}")
    
    classes = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    print(f"Found {len(classes)} Classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(root_folder, class_name)
        count = 0
        for img_name in os.listdir(class_path):
            if max_samples_per_class and count >= max_samples_per_class:
                break
                
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.resize(image_size).convert('RGB')
                    img_array = np.array(img).flatten() / 255.0
                    
                    images.append(img_array)
                    labels.append(class_name)
                    count += 1
            except Exception as e:
                pass
        print(f" -> Loaded {count} images from {class_name}")

    return np.array(images), np.array(labels)

def main():
    print("="*50)
    print(" LAB 5: NN on Blood Cells Classification")
    print("="*50)

    dataset_path = "../dataset/bloodcells"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/bloodcells"

    X, y_labels = load_data(dataset_path, image_size=(32, 32), max_samples_per_class=200)

    if len(X) == 0:
        print("Error: No data loaded.")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nFinal Split: {X_train.shape[0]} Train, {X_test.shape[0]} Test samples")

    print("\n" + "="*50)
    print(" Experiment 1: Compare Network Sizes")
    print("=" * 50)
    
    layers_test = [1, 2, 5]
    nodes_test = [10, 100, 500]
    
    results_size = []
    
    for l in layers_test:
        for n in nodes_test:
            hidden_layers = tuple([n] * l)
            print(f"Training: {l} Layers, {n} Nodes...")
            
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=200, random_state=42)
            mlp.fit(X_train, y_train)
            acc = mlp.score(X_test, y_test)
            
            results_size.append({'Layers': l, 'Nodes': n, 'Accuracy': acc})
            
    df_size = pd.DataFrame(results_size)
    pivot_size = df_size.pivot(index='Layers', columns='Nodes', values='Accuracy')
    print("\nNetwork Size Results:")
    print(pivot_size)

    print("\n" + "="*50)
    print(" Experiment 2: Compare Learning Rates")
    print("=" * 50)
    
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    
    fixed_struct = (100, 100)
    
    results_lr = []
    best_model = None
    best_acc = 0
    
    for lr in learning_rates:
        print(f"Training LR: {lr}...")
        mlp = MLPClassifier(hidden_layer_sizes=fixed_struct, learning_rate_init=lr, max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        
        results_lr.append({
            'Learning Rate': lr, 
            'LR Sci': f"10^{int(np.log10(lr))}",
            'Accuracy': acc
        })
        
        if acc >= best_acc:
            best_acc = acc
            best_model = mlp
            
    df_lr = pd.DataFrame(results_lr)[['LR Sci', 'Learning Rate', 'Accuracy']]
    print("\nLearning Rate Results:")
    print(df_lr)

    print("\nDisplaying samples from Best Model...")
    
    indices = np.random.choice(len(X_test), 10, replace=False)
    plt.figure(figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        img_show = X_test[idx].reshape(32, 32, 3) 
        
        true_label = le.inverse_transform([y_test[idx]])[0]
        pred_label = le.inverse_transform([best_model.predict([X_test[idx]])])[0]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img_show)
        plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
