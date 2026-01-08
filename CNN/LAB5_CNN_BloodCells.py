import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ตั้งค่า GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_dataset_path():
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, '..', 'dataset', 'bloodcells')
    
    if not os.path.exists(dataset_dir):
        # Fallback
        dataset_dir = 'dataset/bloodcells'
        
    return dataset_dir

def load_data(img_size=(150, 150), batch_size=32):
    dataset_dir = get_dataset_path()
    print(f"Loading data from: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory not found!")
        return None, None, None

    # Load and split using validation_split
    # Subset 'training' -> Train set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2, # 20% for test
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Subset 'validation' -> Test set (in this context)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, test_ds, class_names

def build_cnn_model(num_layers, num_filters, input_shape, num_classes, learning_rate=0.001):
    model = models.Sequential()
    
    # Rescaling
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        else:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
            
        if i % 2 == 0:
            model.add(layers.MaxPooling2D((2, 2)))
            
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    IMG_SIZE = (150, 150)
    BATCH_SIZE = 32
    
    train_ds, test_ds, class_names = load_data(IMG_SIZE, BATCH_SIZE)
    
    if train_ds is None: return
    
    input_shape = IMG_SIZE + (3,)
    num_classes = len(class_names)
    
    # --- Experiment 1: Learning Rates ---
    print("\n" + "="*50)
    print(" Experiment 1: Different Learning Rates")
    print(" (Fixed Architecture: 2 Layers, 32 Filters)")
    print("="*50)
    
    lrs_to_test = [0.01, 0.001, 0.0001, 0.00001]
    results_lr = []
    
    for lr in lrs_to_test:
        print(f"Testing Learning Rate: {lr}")
        model = build_cnn_model(num_layers=2, num_filters=32, input_shape=input_shape, num_classes=num_classes, learning_rate=lr)
        
        mode = model.fit(train_ds, epochs=3, verbose=1) # 3 epochs for speed
        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f" -> Accuracy: {acc:.4f}")
        results_lr.append({'Learning Rate': lr, 'Accuracy': acc})

    df_lr = pd.DataFrame(results_lr)
    print("\nResults Table (Learning Rate):")
    print(df_lr)

    # --- Experiment 2: Network Sizes ---
    print("\n" + "="*50)
    print(" Experiment 2: Network Sizes (Layers & Nodes)")
    print(" (Fixed Learning Rate: 0.001)")
    print("="*50)
    
    layers_to_test = [1, 2, 3]
    filters_to_test = [32, 64]
    results_size = []
    
    best_model = None
    best_acc = 0
    
    for l in layers_to_test:
        for f in filters_to_test:
            print(f"\nTesting: {l} Layers, {f} Filters...")
            try:
                model = build_cnn_model(num_layers=l, num_filters=f, input_shape=input_shape, num_classes=num_classes)
                model.fit(train_ds, epochs=3, verbose=1)
                
                loss, acc = model.evaluate(test_ds, verbose=0)
                results_size.append({'Layers': l, 'Filters': f, 'Accuracy': acc})
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
            except Exception as e:
                print(f"Error: {e}")

    df_size = pd.DataFrame(results_size)
    if not df_size.empty:
        pivot_size = df_size.pivot(index='Layers', columns='Filters', values='Accuracy')
        print("\nResults Table (Network Sizes):")
        print(pivot_size)
    print("="*50)
    
    if best_model:
        print("\nSample Predictions from Best Model:")
        plt.figure(figsize=(15, 5))
        for images, labels in test_ds.take(1):
            predictions = best_model.predict(images)
            for i in range(5):
                ax = plt.subplot(1, 5, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                
                pred_label = class_names[np.argmax(predictions[i])]
                true_label = class_names[np.argmax(labels[i])]
                
                color = 'green' if pred_label == true_label else 'red'
                plt.title(f"P:{pred_label}\nT:{true_label}", color=color)
                plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
