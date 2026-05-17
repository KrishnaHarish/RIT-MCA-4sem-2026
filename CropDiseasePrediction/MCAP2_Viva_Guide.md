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