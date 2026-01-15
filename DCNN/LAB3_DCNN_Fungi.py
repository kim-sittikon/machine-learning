import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
import pandas as pd
import os

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Config
DATASET_PATH = r"C:\Users\YEDHEE\Desktop\machine learning\Set Load\Microscopic Fungi Image - DeFungi Dataset"
IMAGE_SIZES = [(50, 50), (150, 150)]
BATCH_SIZE = 32
EPOCHS = 3
MODELS_LIST = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']

def load_data(img_size):
    print(f"\n[INFO] Loading data for size {img_size} from {DATASET_PATH}...")
    
    train_dir = os.path.join(DATASET_PATH, 'train')
    valid_dir = os.path.join(DATASET_PATH, 'valid')
    test_dir = os.path.join(DATASET_PATH, 'test')

    # Load datasets using image_dataset_from_directory
    # This returns a tf.data.Dataset object which is efficient
    train_ds = image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=True
    )
    val_ds = image_dataset_from_directory(
        valid_dir, image_size=img_size, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False
    )
    test_ds = image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def build_model(model_name, input_shape, n_classes):
    print(f"[INFO] Building {model_name}...")
    
    # Rescaling layer meant to be inside model or preprocessing, but since we didn't rescale in dataset loading (defaults to float32 not scaled or int?), 
    # image_dataset_from_directory returns float32 but not scaled 0-1 if not specified. 
    # VGG/ResNet usually expect 0-255 with specific preprocessing, but generic 0-1 or -1 to 1 works too.
    # Let's add a Rescaling layer at the start of the model to be safe and portable.
    
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inp) # Normalize 0-1

    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Transfer Learning: Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x_out = base_model.output
    x_out = GlobalAveragePooling2D()(x_out)
    x_out = Dense(1024, activation='relu')(x_out)
    predictions = Dense(n_classes, activation='softmax')(x_out)
    
    model = Model(inputs=inp, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, test_ds):
    # Get true labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return acc, prec, rec, y_true, y_pred

def main():
    results = []
    
    for size in IMAGE_SIZES:
        print(f"\n{'='*40}")
        print(f"Processing Image Size: {size}")
        print(f"{'='*40}")
        
        train_ds, val_ds, test_ds, class_names = load_data(size)
        input_shape = (size[0], size[1], 3)
        n_classes = len(class_names)
        
        for model_name in MODELS_LIST:
            tf.keras.backend.clear_session()
            
            try:
                model = build_model(model_name, input_shape, n_classes)
                
                print(f"Training {model_name} on {size}...")
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    verbose=1
                )
                
                # Evaluate
                print(f"Evaluating {model_name}...")
                acc, prec, rec, y_true, y_pred = evaluate_model(model, test_ds)
                
                # Metric on Train/Val from history (last epoch)
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                
                # Save Model
                save_dir = os.path.join(os.path.dirname(__file__), 'models')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"model_fungi_{model_name}_{size[0]}x{size[1]}.h5")
                model.save(save_path)
                print(f"Model saved to {save_path}")
                
                results.append({
                    'Image Size': f"{size[0]}x{size[1]}",
                    'Model': model_name,
                    'Train Acc': train_acc,
                    'Val Acc': val_acc,
                    'Test Acc': acc,
                    'Precision': prec,
                    'Recall': rec
                })
                
                # Sample Predictions for the last model of each size (or just the very last one)
                if model_name == 'VGG16' and size == (150, 150):
                   show_sample_predictions(model, test_ds, class_names)

            except Exception as e:
                print(f"Failed to train {model_name} on {size}: {e}")

    # Output Comparison Table
    print(f"\n{'='*20} PERFORMANCE COMPARISON LAB3 {'='*20}")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Analysis
    print(f"\n{'='*20} ANALYSIS: Fungi Recognition vs Classification {'='*20}")
    print("""
    Analysis:
    - Fungi Classification: This is what we did here. We classify an image into one of N predefined fungi classes (species).
      It's a standard multiclass classification problem.
      
    - Fungi "Recognition" (in a broader sense) might imply:
       1. Detecting if fungi is present or not (Detection).
       2. Or identifying a specific instance or strain in a way similar to face recognition (using embeddings),
          but typically in biology, we just call it Classification (identifying the species/genus).
          
    Therefore, the primary task here is Classification which maps input features (images) to discrete labels (Species).
    """)

def show_sample_predictions(model, test_ds, class_names):
    print("\n[INFO] Displaying sample predictions...")
    plt.figure(figsize=(12, 6))
    
    # Take one batch
    for images, labels in test_ds.take(1):
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(labels, axis=1)
        
        for i in range(min(5, len(images))):
            ax = plt.subplot(1, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
