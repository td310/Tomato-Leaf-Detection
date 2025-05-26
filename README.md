# Tomato Leaf Disease Classification with GradCAM

## Project Overview
<p align="justify">This project focuses on classifying tomato leaf diseases using deep learning models, specifically DenseNet121, trained on the **PlantVillage** dataset. The model is trained and evaluated on **Google Colab** to leverage its computational resources, including GPU acceleration. After training, **GradCAM (Gradient-weighted Class Activation Mapping)** is employed to visualize the regions of the input images that contribute most to the model's predictions, providing interpretability for the classification results.</p>

<p align="justify">The dataset used is a cleaned version of the PlantVillage dataset, specifically tailored for tomato leaf diseases (`clean_tomato_dataset_td`). The project includes two main phases: training a deep learning model for disease classification and generating GradCAM heatmaps to highlight important image regions for each predicted class.</p>

## Technologies
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Model Architecture**: DenseNet121, ResNet50, InceptionV3 (pre-trained on ImageNet, fine-tuned for tomato leaf disease classification)
- **Data Augmentation**: ImageDataGenerator (TensorFlow) for rotation, zoom, flip, and shift
- **Visualization**: Matplotlib for plotting GradCAM heatmaps
- **GradCAM**: Custom implementation to generate class activation heatmaps
- **Environment**: Google Colab for training and inference
- **Libraries**: NumPy, OpenCV (cv2), Collections
- **Dataset**: PlantVillage (cleaned tomato leaf dataset)

## Training
### Model Training
The project trains deep learning models (DenseNet121, ResNet50, and InceptionV3) to classify tomato leaf diseases. The training process involves:
- **Pre-trained Models**: Models are initialized with ImageNet weights and fine-tuned on the tomato leaf dataset.
- **Architecture**: Each model is extended with custom layers for classification, including global average pooling, dense layers, dropout for regularization, and a softmax output layer.
- **Training Strategy**: Models are trained in two phases:
  1. Initial training of the top layers with the base model frozen to adapt to the dataset.
  2. Fine-tuning of the entire model with a lower learning rate to improve performance.
- **Optimization**: Uses the Adam optimizer with categorical cross-entropy loss and appropriate learning rate scheduling.
- **Callbacks**: Includes checkpoints to save the best model weights, learning rate reduction on plateau, and early stopping to prevent overfitting.
- **Output**: Trained models and weights are saved to Google Drive for later use.

### Data Preparation
- **Dataset**: The PlantVillage dataset is organized into `train` and `val` directories.
- **Data Augmentation**: Applied to the training set to enhance generalization, including rescaling, rotation, width/height shift, zoom, and horizontal flip.
- **Validation Data**: Only rescaled without augmentation.
- **Image Size**: 224x224 pixels
- **Batch Size**: 32

### GradCAM Implementation
- **Purpose**: GradCAM visualizes the regions of input images that influence the model's predictions.
- **Methodology**:
  - The last convolutional layer of each model (e.g., `conv5_block16_concat` for DenseNet121, `conv5_block3_out` for ResNet50, `mixed10` for InceptionV3) is used to compute gradients.
  - The `make_gradcam_heatmap` function generates a heatmap by:
    1. Creating a gradient model to output convolutional layer activations and predictions.
    2. Computing gradients of the predicted class score with respect to the convolutional layer's output.
    3. Pooling gradients and weighting convolutional outputs to create a normalized heatmap.
  - The `overlay_heatmap` function overlays the heatmap on the original image using OpenCV with a JET colormap and alpha blending (0.4).

## GradCAM Result
- ResNet50
![image](https://github.com/user-attachments/assets/e04f63c7-a8cc-45ae-8c0e-eae3cab2f5a0)
- InceptionV3
![image](https://github.com/user-attachments/assets/04ebb0e2-f660-491b-a70a-f62e9dcf8289)
- DenseNet121
![image](https://github.com/user-attachments/assets/5f090c46-c3de-44f3-88d4-5a11d5eaf658)


