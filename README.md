# AgroVision-CNN – Vegetable Classification Using CNN + TensorFlow Lite

AgroVision-CNN is a deep learning project focused on classifying vegetable images using a custom-built **Convolutional Neural Network (CNN)**.  
The trained model is exported to **TensorFlow Lite** (`.tflite`) for lightweight deployment on mobile or embedded systems.

This project demonstrates a complete ML workflow: dataset loading → preprocessing → CNN training → evaluation → TFLite conversion.

---

## 🚀 Features

- 🧠 Custom CNN model built using TensorFlow/Keras  
- 🥕 Classifies multiple vegetable types  
- 💾 Saved trained model (`.h5`)  
- 🔄 Exported TFLite model for mobile/edge deployment  
- 📓 Includes complete Jupyter Notebook with step-by-step pipeline  
- 📊 High accuracy and lightweight model architecture  

---

## 📂 Project Structure

project/
│
├── CNN_model.ipynb # Full training notebook
├── vegetables.h5 # Trained CNN model
├── tf_lite_vegetable.tflite # TFLite converted model
└── README.md # Documentation

---

## 🧠 Model Development

The model is developed using a custom CNN architecture with:

- Conv2D layers  
- MaxPooling  
- Batch Normalization  
- Dense (fully connected) layers  
- Softmax output layer  

### Training Steps

1. Load vegetable images dataset  
2. Resize and normalize images  
3. Build CNN architecture  
4. Train model (fit, train/val split)  
5. Evaluate accuracy and loss  
6. Save trained model (`vegetables.h5`)  
7. Convert to TFLite format (`tf_lite_vegetable.tflite`)

---

## 🔧 Conversion to TensorFlow Lite

The trained model was converted using:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("tf_lite_vegetable.tflite", "wb") as f:
    f.write(tflite_model)
