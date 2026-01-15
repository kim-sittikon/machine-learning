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
import cv2

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Config
DATASET_PATH = r"C:\Users\YEDHEE\Desktop\machine learning\Set Load\Covid19-dataset"
IMAGE_SIZES = [(120, 120), (224, 224)]
BATCH_SIZE = 32
EPOCHS = 3
MODELS_LIST = ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2']

def load_data(img_size):
    print(f"\n[INFO] Loading data for size {img_size} from {DATASET_PATH}...")
    
    train_dir = os.path.join(DATASET_PATH, 'train')
    test_dir = os.path.join(DATASET_PATH, 'test')

    # Create Train and Validation split from 'train' folder
    # 80% Train, 20% Validation
    train_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False # Don't shuffle test for consistent evaluation
    )
    
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def build_model(model_name, input_shape, n_classes):
    print(f"[INFO] Building {model_name}...")
    
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inp)

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

    # Transfer Learning
    for layer in base_model.layers:
        layer.trainable = False
        
    x_out = base_model.output
    x_out = GlobalAveragePooling2D()(x_out)
    x_out = Dense(1024, activation='relu')(x_out)
    predictions = Dense(n_classes, activation='softmax')(x_out)
    
    model = Model(inputs=inp, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_last_conv_layer_name(model_name):
    if model_name == 'VGG16':
        return 'block5_conv3'
    elif model_name == 'ResNet50':
        return 'conv5_block3_out'
    elif model_name == 'DenseNet121':
        return 'relu' 
    elif model_name == 'MobileNetV2':
        return 'Out_relu'
    return None

def evaluate_and_gradcam(model, test_ds, model_name, size, class_names):
    # Metrics
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

    return acc, prec, rec

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
                
                print(f"Evaluating {model_name}...")
                acc, prec, rec = evaluate_and_gradcam(model, test_ds, model_name, size, class_names)
                
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                
                # Save Model
                save_dir = os.path.join(os.path.dirname(__file__), 'models')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"model_covid_{model_name}_{size[0]}x{size[1]}.h5")
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
                
                # Sample Predictions & Heatmap (Simplified)
                # Show predictions for the best looking model/size combo (e.g., VGG 224x224)
                if model_name == 'VGG16' and size == (224, 224):
                    print(f"\n[INFO] Displaying sample predictions for {model_name}...")
                    plt.figure(figsize=(12, 6))
                    for images, labels in test_ds.take(1):
                        preds = model.predict(images, verbose=0)
                        pred_labels = np.argmax(preds, axis=1)
                        true_labels = np.argmax(labels, axis=1)
                        
                        for i in range(min(5, len(images))):
                            ax = plt.subplot(1, 5, i + 1)
                            # Display Image
                            plt.imshow(images[i].numpy().astype("uint8"))
                            
                            # Title
                            title = f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}"
                            plt.title(title)
                            plt.axis("off")
                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                print(f"Failed to train {model_name} on {size}: {e}")

    # Output Comparison Table
    print(f"\n{'='*20} PERFORMANCE COMPARISON LAB5 {'='*20}")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Analysis
    print(f"\n{'='*20} ANALYSIS: COVID-19 Recognition vs Classification {'='*20}")
    print("""
    Analysis:
    - COVID-19 Classification: Categorizing X-ray/CT images into 'Covid', 'Normal', or 'Pneumonia'.
    - COVID-19 Recognition: Often implies identifying specific patterns or severity (Recognition of features).
    - In medical imaging, 'Classification' is the diagnosis (Disease vs Healthy).
    - 'Detection' or 'Segmentation' would be finding the exact area of infection.
    - This Lab performs Classification of X-ray images.
    """)

if __name__ == "__main__":
    main()
