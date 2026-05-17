# MCAP2 Viva Guide: Tomato Disease Prediction

## 1. Project Summary
This project is a tomato leaf disease prediction system built with PyTorch and Streamlit. It classifies an input leaf image into one of 10 tomato disease classes or healthy.

## 2. Why It Fits MCAP2
This is a suitable Phase II project because it includes:
- a clear real-world problem statement
- dataset handling and preprocessing
- model training and evaluation
- inference and a user-facing demo
- documentation of metrics and limitations

## 3. 20-Minute Viva Question Set

### Core Questions
1. What is the objective of the project?
2. Why did you choose tomato disease prediction?
3. Which dataset did you use?
4. How many classes are being predicted?
5. Which model architecture did you choose and why?
6. What is transfer learning?
7. Why did you use ResNet18?
8. How does the training pipeline work?
9. What preprocessing is applied to the images?
10. Which loss function and optimizer are used?

### Evaluation and Results
11. Why are accuracy alone not enough for evaluation?
12. What do precision, recall, and F1-score measure?
13. What does the confusion matrix show?
14. What were the current benchmark results?
15. Why is the 1-epoch baseline not considered the final result?

### Implementation and Demo
16. How does the prediction script work?
17. What does the Streamlit app demonstrate?
18. How is the model artifact stored and loaded?
19. What are the main limitations of the project?
20. What future improvements would you suggest?

## 4. Short Answers for Rapid Revision

**Q: What is the project about?**  
A: It predicts tomato disease class from a leaf image using a deep learning model.

**Q: Why is it useful?**  
A: It supports early disease detection, which can help farmers take timely action.

**Q: Why ResNet18?**  
A: It is lightweight, reliable, and works well for transfer learning.

**Q: What metrics did you use?**  
A: Accuracy, precision, recall, F1-score, and confusion matrix.

**Q: What is the main limitation?**  
A: The quick baseline run is only a proof of pipeline execution and needs stronger training for final performance.

## 5. Suggested Viva Opening
"My project is a tomato leaf disease prediction system. It uses deep learning to classify a leaf image into one of the tomato disease categories, and it also includes training, evaluation, and a simple demo interface."

## 6. Suggested Closing Line
"This project demonstrates a complete applied machine learning pipeline from data preparation to inference, which makes it suitable for an MCA Phase II submission."

Below is a **complete 20-minute viva preparation guide** for your **Tomato/Crop Disease Prediction using AI** project.

You can revise this directly before viva.

***

# Complete 20-Minute Viva Guide

## Project: Tomato Disease Prediction Using Deep Learning

***

## 1. Viva Opening — 1 Minute

You can start like this:

> Good morning. My project is **Tomato Disease Prediction using Deep Learning**.  
> The objective of this project is to classify tomato leaf images into healthy or diseased categories.  
> I used a **ResNet18 deep learning model** with **transfer learning**, implemented using **PyTorch**.  
> The model takes a tomato leaf image as input and predicts the disease class.  
> I also created a simple **Streamlit web application** where the user can upload a leaf image and get the prediction result.  
> This project covers the full machine learning pipeline, including dataset preparation, preprocessing, model training, evaluation, prediction, and demo deployment.

***

# 2. Project Summary Questions

## Q1. What is your project about?

**Answer:**

> My project is about predicting tomato leaf diseases using deep learning. The system takes a tomato leaf image as input and classifies it into one of the disease or healthy categories.

***

## Q2. What is the objective of your project?

**Answer:**

> The main objective is to build an AI-based system that can identify tomato leaf diseases from images. This can help in early disease detection and support farmers in taking timely action.

***

## Q3. Why did you choose this project?

**Answer:**

> I chose this project because agriculture is very important, and crop diseases can reduce yield and quality. Tomato is a commonly grown crop, and detecting diseases early can help farmers reduce losses. Also, this project allowed me to apply deep learning to a real-world problem.

***

## Q4. What problem does your project solve?

**Answer:**

> It solves the problem of manual disease identification from tomato leaves. Farmers may not always have immediate access to experts, so an AI-based image classification system can assist in identifying the disease quickly.

***

## Q5. Is this project useful in real life?

**Answer:**

> Yes, it is useful as an early disease detection support system. However, in real-world usage, it should be combined with expert agricultural advice because image-based prediction may not always be 100% accurate.

***

# 3. Dataset Questions

## Q6. Which dataset did you use?

**Answer:**

> I used a tomato leaf disease image dataset, commonly organized similar to the PlantVillage dataset structure. It contains images of tomato leaves belonging to different disease categories and healthy leaves.

***

## Q7. How many classes are predicted?

**Answer:**

> The model predicts **10 classes total**, including tomato disease classes and one healthy class.

A safer answer:

> In my project, the number of output classes is based on the folders present in the dataset. For the common tomato dataset, there are 10 classes including healthy.

***

## Q8. What are the common classes in tomato disease prediction?

**Answer:**

Common tomato leaf classes are:

1. Bacterial spot
2. Early blight
3. Late blight
4. Leaf mold
5. Septoria leaf spot
6. Spider mites
7. Target spot
8. Tomato mosaic virus
9. Tomato yellow leaf curl virus
10. Healthy

***

## Q9. How is the dataset organized?

**Answer:**

> The dataset is organized into folders, where each folder represents one class. Images inside each folder belong to that disease or healthy category. This structure makes it easy to load the dataset using image classification utilities.

Example:

```text
dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── Tomato___healthy/
│
└── val/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    └── Tomato___healthy/
```

***

## Q10. Why do we split the dataset into training and testing/validation?

**Answer:**

> The training set is used to train the model, while the validation or test set is used to check how well the model performs on unseen images. This helps us know whether the model has actually learned or is only memorizing the training images.

***

# 4. AI Model Questions

## Q11. Which AI model did you use?

**Answer:**

> I used **ResNet18**, which is a pretrained Convolutional Neural Network model. It is used for image classification tasks.

***

## Q12. What type of model is ResNet18?

**Answer:**

> ResNet18 is a deep learning model based on Convolutional Neural Networks. It uses residual connections to improve training and avoid the vanishing gradient problem.

***

## Q13. Why did you use ResNet18?

**Answer:**

> I used ResNet18 because it is lightweight, faster to train, and suitable for transfer learning. It gives good performance for image classification without requiring very high computational resources.

***

## Q14. What is CNN?

**Answer:**

> CNN stands for Convolutional Neural Network. It is a type of deep learning model mainly used for image processing and classification. CNNs can automatically learn features such as edges, colors, textures, and patterns from images.

***

## Q15. What is transfer learning?

**Answer:**

> Transfer learning means using a model that has already been trained on a large dataset and adapting it for a new task. In my project, I used a pretrained ResNet18 model and modified its final layer for tomato disease classification.

***

## Q16. Why is transfer learning useful?

**Answer:**

> Transfer learning is useful because it saves training time and gives better results when the dataset is limited. The pretrained model already knows general image features, so we only need to fine-tune it for tomato leaf disease classification.

***

## Q17. What changes did you make in ResNet18?

**Answer:**

> I replaced the final fully connected classification layer of ResNet18 with a new layer matching the number of tomato disease classes.

Example explanation:

```text
Original ResNet18 output: 1000 classes
Modified output: 10 tomato classes
```

***

## Q18. What is the input image size?

**Answer:**

> The input images are resized to **224 × 224 pixels**, because ResNet18 commonly uses this input size.

***

# 5. Training Questions

## Q19. How does your training pipeline work?

**Answer:**

> First, the dataset is loaded and preprocessing is applied. Then the ResNet18 model is loaded with pretrained weights. The final layer is modified according to the number of classes. After that, the model is trained using the training data, validated using validation data, and finally the trained model is saved as a model file.

***

## Q20. What preprocessing is applied?

**Answer:**

> The images are resized, converted into tensors, and normalized. Data augmentation may also be applied during training, such as random rotation, flipping, or color adjustment, to improve model generalization.

***

## Q21. Why do we resize images?

**Answer:**

> Images are resized so that all inputs have the same size and can be passed into the neural network. ResNet18 expects a fixed image size, commonly 224 × 224.

***

## Q22. Why do we normalize images?

**Answer:**

> Normalization scales pixel values into a standard range. This helps the model train more smoothly and improves convergence.

***

## Q23. What is data augmentation?

**Answer:**

> Data augmentation means creating variations of training images using transformations like rotation, flipping, cropping, or brightness changes. It helps the model become more robust and reduces overfitting.

***

## Q24. Which loss function did you use?

**Answer:**

> I used **Cross Entropy Loss** because this is a multi-class classification problem.

***

## Q25. Why Cross Entropy Loss?

**Answer:**

> Cross Entropy Loss is suitable for multi-class classification. It compares the predicted class probabilities with the actual class label and calculates the error. The model tries to reduce this error during training.

***

## Q26. Which optimizer did you use?

**Answer:**

> I used the **Adam optimizer** to update the model weights during training.

***

## Q27. Why Adam optimizer?

**Answer:**

> Adam optimizer is efficient and commonly used in deep learning. It adjusts the learning rate automatically for each parameter and usually converges faster than basic gradient descent.

***

## Q28. What is an epoch?

**Answer:**

> One epoch means the model has seen the complete training dataset once during training.

***

## Q29. What is batch size?

**Answer:**

> Batch size is the number of images processed at one time before updating the model weights.

***

## Q30. What is learning rate?

**Answer:**

> Learning rate controls how much the model weights are updated during training. If it is too high, the model may not learn properly. If it is too low, training becomes slow.

***

# 6. Evaluation Questions

## Q31. Which evaluation metrics did you use?

**Answer:**

> I used accuracy, precision, recall, F1-score, and confusion matrix to evaluate the model.

***

## Q32. What is accuracy?

**Answer:**

> Accuracy is the percentage of correctly predicted images out of the total images.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

***

## Q33. Why is accuracy alone not enough?

**Answer:**

> Accuracy alone may be misleading if the dataset is imbalanced. A model may perform well on majority classes but poorly on minority classes. That is why precision, recall, F1-score, and confusion matrix are also important.

***

## Q34. What is precision?

**Answer:**

> Precision tells how many predicted positive results are actually correct.

Simple explanation:

> If the model predicts a disease, precision tells how often that prediction is correct.

***

## Q35. What is recall?

**Answer:**

> Recall tells how many actual disease cases were correctly identified by the model.

Simple explanation:

> If a disease is actually present, recall tells whether the model successfully detected it.

***

## Q36. What is F1-score?

**Answer:**

> F1-score is the balance between precision and recall. It is useful when we want a combined performance measure.

***

## Q37. What is a confusion matrix?

**Answer:**

> A confusion matrix shows correct and incorrect predictions for each class. It helps identify which disease classes are being confused by the model.

***

## Q38. What does the confusion matrix help you understand?

**Answer:**

> It helps us understand which classes are predicted correctly and which classes are misclassified. For example, the model may confuse Early Blight and Late Blight because their symptoms may look similar.

***

## Q39. Why is a 1-epoch result not final?

**Answer:**

> A 1-epoch run is only a quick baseline to check whether the pipeline works. The model needs more epochs and tuning to learn better features and achieve stronger performance.

***

## Q40. What is overfitting?

**Answer:**

> Overfitting happens when the model performs very well on training data but poorly on unseen test data. It means the model has memorized the training images instead of learning general patterns.

***

## Q41. How can overfitting be reduced?

**Answer:**

Overfitting can be reduced by:

* Using data augmentation
* Increasing dataset size
* Using dropout
* Using regularization
* Early stopping
* Validating on unseen data

***

# 7. Prediction and Streamlit Demo Questions

## Q42. How does the prediction work?

**Answer:**

> The prediction script loads the trained model, applies the same preprocessing to the uploaded image, passes the image through the model, and selects the class with the highest probability as the predicted result.

***

## Q43. What is model inference?

**Answer:**

> Inference means using the trained model to make predictions on new input data.

***

## Q44. How is the model saved?

**Answer:**

> The trained PyTorch model weights are saved in a `.pth` file. This file can be loaded later for prediction without retraining the model.

***

## Q45. Why do we use `model.eval()`?

**Answer:**

> `model.eval()` sets the model to evaluation mode. It disables training-specific behavior like dropout and batch normalization updates, which helps produce stable predictions during inference.

***

## Q46. What does the Streamlit app do?

**Answer:**

> The Streamlit app provides a simple user interface where the user can upload a tomato leaf image. The app processes the image, sends it to the trained model, and displays the predicted disease class.

***

## Q47. Why did you use Streamlit?

**Answer:**

> I used Streamlit because it is simple and fast for building machine learning demo applications. It allows users to interact with the model without using the command line.

***

## Q48. How does a user use your app?

**Answer:**

> The user opens the Streamlit app, uploads a tomato leaf image, and clicks or waits for prediction. The app then displays the predicted class name.

***

# 8. GitHub and Project Structure Questions

## Q49. Where is your project stored?

**Answer:**

> My project is stored in a GitHub repository under the CropDiseasePrediction folder.

***

## Q50. Why did you use GitHub?

**Answer:**

> I used GitHub for version control, backup, project organization, and sharing the code for review.

***

## Q51. What files are present in your project?

**Answer:**

> The project contains files for training, prediction, evaluation, requirements, model storage, and the Streamlit application.

Example:

```text
CropDiseasePrediction/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── README.md
├── models/
└── outputs/
```

***

# 9. Limitations and Future Scope

## Q52. What are the limitations of your project?

**Answer:**

> The main limitation is that the model depends on image quality and dataset diversity. If the uploaded image has poor lighting, blur, or different field conditions, accuracy may reduce. Also, a short training run is only a baseline and not the final optimized result.

***

## Q53. Can this model work in real farms?

**Answer:**

> It can be used as a support tool, but for real farm deployment, it needs more real-world images, better testing, expert validation, and possibly mobile-based deployment.

***

## Q54. What future improvements can be added?

**Answer:**

Future improvements include:

* Training for more epochs
* Using larger real-world datasets
* Improving data augmentation
* Trying models like ResNet50, EfficientNet, or MobileNet
* Adding confidence score
* Adding treatment suggestions
* Deploying the app online
* Creating a mobile app version
* Supporting more crops

***

## Q55. What is the conclusion of your project?

**Answer:**

> The project successfully demonstrates a complete deep learning pipeline for tomato leaf disease classification. It includes dataset preprocessing, training using ResNet18, evaluation, prediction, and a Streamlit demo. It is a useful academic prototype for AI-based crop disease detection.

***

# 10. 20-Minute Viva Flow

Use this order in viva if they ask you to explain the complete project.

## Minute 0–2: Introduction

Say:

> My project is Tomato Disease Prediction using Deep Learning. It classifies tomato leaf images into disease or healthy categories. I used ResNet18 with transfer learning in PyTorch and created a Streamlit app for demonstration.

***

## Minute 2–4: Problem Statement

Say:

> Farmers may face difficulty identifying diseases early. Delay in disease detection can reduce crop yield. This project provides an AI-based image classification system to support early disease detection.

***

## Minute 4–6: Dataset

Say:

> The dataset contains tomato leaf images arranged in class-wise folders. Each folder represents one disease or healthy class. The images are split into training and validation/testing data.

***

## Minute 6–9: Model

Say:

> I used ResNet18, a CNN-based deep learning model. Since it is pretrained, I used transfer learning. I replaced the final layer according to the number of tomato classes.

***

## Minute 9–12: Training

Say:

> The images are resized, converted to tensors, and normalized. The model is trained using Cross Entropy Loss and Adam optimizer. The training process updates model weights to reduce classification error.

***

## Minute 12–15: Evaluation

Say:

> I evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix. Accuracy gives overall correctness, while precision, recall, and F1-score give class-level performance. The confusion matrix shows where the model is making mistakes.

***

## Minute 15–17: Demo

Say:

> In the Streamlit app, the user uploads a tomato leaf image. The image is preprocessed and passed to the trained model. The model predicts the most likely disease class and displays the result.

***

## Minute 17–19: Limitations and Future Scope

Say:

> The project is currently an academic prototype. Its performance depends on dataset quality and image conditions. In the future, I can improve it by training for more epochs, using more real-world images, adding confidence scores, treatment suggestions, and deploying it as a mobile app.

***

## Minute 19–20: Closing

Say:

> Overall, this project demonstrates a complete applied machine learning pipeline from data preparation to model training, evaluation, inference, and user-facing demo. It is suitable for MCA Phase II because it solves a real-world problem using AI and deep learning.

***

# 11. Important One-Line Answers

## Model used

> ResNet18 pretrained CNN model.

## AI technique used

> Deep learning with transfer learning.

## Framework used

> PyTorch.

## Demo tool used

> Streamlit.

## Loss function

> Cross Entropy Loss.

## Optimizer

> Adam optimizer.

## Task type

> Multi-class image classification.

## Input

> Tomato leaf image.

## Output

> Predicted disease or healthy class.

## Main metric

> Accuracy, precision, recall, F1-score, and confusion matrix.

## Main limitation

> Performance depends on dataset quality and real-world image conditions.

***

# 12. Final Short Viva Script

Memorize this:

> My project is Tomato Disease Prediction using Deep Learning. It uses tomato leaf images to predict whether the leaf is healthy or affected by a disease. I used the ResNet18 pretrained CNN model with transfer learning. The final layer of the model was modified according to the number of tomato disease classes.
>
> The dataset images are resized, converted into tensors, normalized, and then given to the model for training. I used Cross Entropy Loss because this is a multi-class classification problem, and Adam optimizer to update the model weights efficiently.
>
> The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix. I also created a Streamlit application where the user can upload a tomato leaf image and get the predicted disease class.
>
> The current version is an academic prototype. In the future, it can be improved using more real-world images, longer training, better model tuning, treatment suggestions, and mobile deployment.

***

# 13. If Examiner Asks “What Is Your Contribution?”

Answer:

> My contribution is building a complete deep learning pipeline for tomato disease prediction. I handled dataset preparation, preprocessing, model training using ResNet18, evaluation, saving/loading the model, and creating a Streamlit-based demo for user interaction.

***

# 14. If Examiner Asks “Why Should We Accept This as Phase II?”

Answer:

> This project is suitable for Phase II because it includes a real-world problem, dataset handling, preprocessing, model training, evaluation, inference, and a working demo interface. It demonstrates both machine learning knowledge and practical implementation skills.

***

## Best Final Closing Line

> This project shows how artificial intelligence can be applied to agriculture for early disease detection, and it demonstrates a complete machine learning workflow from data to deployment.
