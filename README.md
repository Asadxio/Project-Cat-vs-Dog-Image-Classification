# Project-Cat-vs-Dog-Image-Classification

# Project Report: Cat vs Dog Image Classification

## Introduction

The objective of this project is to classify images as either cats or dogs using a Convolutional Neural Network (CNN). The project utilizes a dataset of cat and dog images, split into training and test sets.

The dataset used in this project is stored in Google Drive, with the training images located at "/content/drive/MyDrive/best_model/train" and the test images at "/content/drive/MyDrive/best_model/test".

## Project Steps

### 1. Import Libraries and Mount Google Drive

The necessary libraries and modules are imported, and the Google Drive is mounted to access the dataset.

### 2. Load and Preprocess Images

The training image paths are loaded, and the paths are shuffled randomly. The first five image paths are displayed to verify the loading process. The OpenCV library is used to read and resize the images, and the images are converted to arrays. The class labels are extracted from the image paths.

### 3. Split the Dataset

The dataset is split into training and validation sets using the train_test_split function from scikit-learn. The sizes of the training and validation sets are displayed.

### 4. Data Augmentation

An ImageDataGenerator object is created to apply data augmentation techniques, such as rotation, shifting, shearing, zooming, and flipping, to the training images. This helps to increase the robustness and variability of the training data.

### 5. Create the CNN Model

A CNN model based on the LeNet architecture is defined using the Keras Sequential API. The model consists of convolutional, activation, and pooling layers, followed by fully connected layers. The model is compiled with the Adam optimizer and categorical cross-entropy loss.

### 6. Train the Model

The model is trained using the fit function with the training data and validation data. The number of epochs and batch size are defined. The training progress is displayed during the training process.

### 7. Plot Training and Validation Accuracy

The training and validation accuracy values are plotted against the number of epochs to visualize the model's learning progress.

### 8. Save the Trained Model

The trained model is saved to "/content/drive/MyDrive/best_model/cat_dog_new.model" for future use.

### 9. Test the Model on Test Images

The saved model is loaded, and the test images are processed to make predictions. The predictions are displayed along with the corresponding images. The image numbers and predicted labels are stored in a DataFrame for further analysis.

### 10. Convert Predicted Labels to Numeric Values

The predicted labels are converted from categorical values ("Cat" and "Dog") to numeric values (0 and 1) for analysis purposes.

### 11. Create DataFrame with Image Numbers and Predicted Labels

A DataFrame is created with columns for image numbers and predicted labels.

## Conclusion

In this project, a CNN model was developed to classify images as either cats or dogs. The model was trained using the training dataset and achieved a certain level of accuracy. The trained model was then used to make predictions on the test images, and the results were stored in a DataFrame for analysis.

The project demonstrates the application of CNNs for image classification tasks and provides a foundation for further research and improvements in cat vs dog image classification.
