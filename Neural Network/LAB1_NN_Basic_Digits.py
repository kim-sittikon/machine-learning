import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading digits dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    print("\nTraining a sample model for visualization...")
    sample_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    sample_mlp.fit(X_train, y_train)
    
    y_pred_sample = sample_mlp.predict(X_test)
    sample_accuracy = accuracy_score(y_test, y_pred_sample)
    print(f"Sample Model Accuracy: {sample_accuracy:.4f}")

    print("Displaying sample predictions...")
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap='binary')
        ax.set_title(f"True: {y_test[i]}\nPred: {y_pred_sample[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("\nComparing different network sizes (this may take a while)...")
    
    layers_range = range(1, 11) 
    nodes_range = [10, 50, 100, 500, 1000]
    
    results = []

    for layers in layers_range:
        for nodes in nodes_range:
            hidden_layers = tuple([nodes] * layers)
            
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=200, random_state=42)
            mlp.fit(X_train, y_train)
            
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"Layers: {layers}, Nodes/Layer: {nodes} -> Accuracy: {acc:.4f}")
            
            results.append({
                'Hidden Layers': layers,
                'Nodes per Layer': nodes,
                'Accuracy': acc
            })

    results_df = pd.DataFrame(results)
    
    pivot_table = results_df.pivot(index='Hidden Layers', columns='Nodes per Layer', values='Accuracy')
    
    print("\nComparison Results (Accuracy):")
    print(pivot_table)

if __name__ == "__main__":
    main()
