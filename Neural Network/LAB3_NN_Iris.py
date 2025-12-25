import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("Loading Iris dataset...")
    
    possible_paths = ['dataset/Iris.csv', '../dataset/Iris.csv']
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
            
    if dataset_path is None:
        print("Error: ไม่พบไฟล์ 'dataset/Iris.csv'")
        print("กรุณาสร้างโฟลเดอร์ 'dataset' และใส่ไฟล์ Iris.csv ลงไป หรือแก้ไข path ในโค้ด")
        return

    df = pd.read_csv(dataset_path)

    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    
    hidden_layers_config = (50, 50, 50, 50, 50)
    
    results = []
    models = {}

    print(f"\nTraining NN with 5 Layers {hidden_layers_config}...")
    print(f"{'Learning Rate':<15} | {'Accuracy':<10}")
    print("-" * 30)

    for lr in learning_rates:
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers_config,
            learning_rate_init=lr,
            max_iter=1000,
            random_state=42
        )
        
        mlp.fit(X_train, y_train)
        
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Learning Rate': lr,
            'LR Scientific': f"10^{int(np.log10(lr))}",
            'Accuracy': acc
        })
        
        models[lr] = mlp
        print(f"{lr:<15} | {acc:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df[['LR Scientific', 'Learning Rate', 'Accuracy']]
    
    print("\n" + "="*40)
    print(" Comparison Results (Learning Rate) ")
    print("="*40)
    print(results_df)
    print("="*40)

    best_lr = 0.01 
    best_model = models[best_lr]
    
    print(f"\nSample Predictions (using model with LR={best_lr}):")
    
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test[sample_indices]
    y_sample_true = y_test[sample_indices]
    
    y_sample_pred = best_model.predict(X_sample)
    
    y_true_names = le.inverse_transform(y_sample_true)
    y_pred_names = le.inverse_transform(y_sample_pred)

    for i in range(5):
        status = "✅ Correct" if y_sample_true[i] == y_sample_pred[i] else "❌ Wrong"
        print(f"Sample {i+1}: True=[{y_true_names[i]:<15}] vs Pred=[{y_pred_names[i]:<15}] -> {status}")

if __name__ == "__main__":
    main()
