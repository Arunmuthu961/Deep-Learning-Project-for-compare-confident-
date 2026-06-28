import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import gradio as gr
from PIL import Image

# ==========================================
# 1. SIMULATED DATASET SETUP (For Demo)
# ==========================================
DATASET_DIR = "brain_tumor_dataset"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (150, 150)
BATCH_SIZE = 4
EPOCHS = 1  

def create_mock_dataset():
    """Creates a temporary dummy structure so the code compiles and trains instantly."""
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
        for i in range(4):
            img_array = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(DATASET_DIR, cls, f"fake_{i}.jpg"))

if not os.path.exists(DATASET_DIR):
    print("Creating simulated dataset directories for structural verification...")
    create_mock_dataset()

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. MODEL BUILDING & TRAINING
# ==========================================

def build_custom_cnn():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(150, 150, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  
    model = models.Sequential([
        layers.Rescaling(1./255),  
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n--- Training Custom CNN ---")
cnn_model = build_custom_cnn()
cnn_history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
cnn_eval = cnn_model.evaluate(val_ds, verbose=0)
cnn_accuracy = round(float(cnn_eval[1]) * 100, 2) 

print("\n--- Training ResNet50 ---")
resnet_model = build_resnet50()
resnet_history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
resnet_eval = resnet_model.evaluate(val_ds, verbose=0)
resnet_accuracy = round(float(resnet_eval[1]) * 100, 2)

# ==========================================
# 3. PREDICTION ENGINE & UI
# ==========================================

def predict_tumor(input_image):
    if input_image is None:
        return "No image uploaded", "N/A", "No image uploaded", "N/A", "N/A"
    
    img = Image.fromarray(input_image)
    img = img.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    # Run predictions
    cnn_preds = cnn_model.predict(img_array)[0]
    resnet_preds = resnet_model.predict(img_array)[0]

    # Parse predictions
    cnn_class = CLASSES[np.argmax(cnn_preds)]
    cnn_conf = round(float(np.max(cnn_preds)) * 100, 2)
    
    resnet_class = CLASSES[np.argmax(resnet_preds)]
    resnet_conf = round(float(np.max(resnet_preds)) * 100, 2)

    cnn_status = f"Predicted Class: {cnn_class.upper()} ({cnn_conf}% confidence)"
    resnet_status = f"Predicted Class: {resnet_class.upper()} ({resnet_conf}% confidence)"
    
    # NEW LOGIC: Compare single-scan prediction confidence instead of dataset validation accuracy
    if cnn_conf >= resnet_conf:
        best_pipeline = "Custom CNN Pipeline"
        winning_class = cnn_class.upper()
        winning_conf = cnn_conf
    else:
        best_pipeline = "ResNet50 Pipeline"
        winning_class = resnet_class.upper()
        winning_conf = resnet_conf
        
    comparison_summary = f"Based on this specific report upload, the highest confidence pipeline is the {best_pipeline}, predicting {winning_class} with {winning_conf}% confidence."

    return (
        cnn_status, 
        f"{cnn_accuracy}%", 
        resnet_status, 
        f"{resnet_accuracy}%", 
        comparison_summary
    )

# --- Build Gradio Interface layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Brain Tumor Classifier: Custom CNN vs. ResNet50")
    gr.Markdown("Upload an MRI image report slice below to instantly analyze, compare execution status, and verify accuracy pipelines side by side.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Scan Report (Glioma / Meningioma / Pituitary / No Tumor)")
            submit_btn = gr.Button("Analyze and Compare Models", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Evaluation Dashboard")
            
            with gr.Row():
                with gr.Group():
                    gr.Markdown("#### 🧬 Custom CNN Pipeline")
                    cnn_acc_out = gr.Textbox(label="Model Training Accuracy Status")
                    cnn_pred_out = gr.Textbox(label="Scan Prediction Status")
                    
                with gr.Group():
                    gr.Markdown("#### 🚀 ResNet50 Pipeline")
                    resnet_acc_out = gr.Textbox(label="Model Training Accuracy Status")
                    resnet_pred_out = gr.Textbox(label="Scan Prediction Status")
            
            # Label changed to reflect per-scan confidence evaluation
            summary_out = gr.Textbox(label="🏆 Per-Scan Highest Confidence Conclusion", lines=2)

    submit_btn.click(
        fn=predict_tumor,
        inputs=image_input,
        outputs=[cnn_pred_out, cnn_acc_out, resnet_pred_out, resnet_acc_out, summary_out]
    )

if __name__ == "__main__":
    demo.launch()
