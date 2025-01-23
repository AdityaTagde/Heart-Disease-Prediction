# 🫀 Heart Disease Prediction 🫀

This project aims to predict the presence of heart disease in patients based on various medical features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic results, maximum heart rate, exercise-induced angina, ST depression induced by exercise, and several other features. The prediction model is built using machine learning algorithms, and the best-performing model is saved for later use. 🫀

## 📋 Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Dataset](#dataset)
- [Steps](#steps)
- [Model](#model)
- [Usage](#usage)
- [License](#license)

## 📝 Description

This project uses machine learning to predict whether a patient is likely to have heart disease based on a set of health-related features. The dataset is split into training and test sets, and multiple models are evaluated for accuracy. The model with the best performance is selected and saved for future predictions. Early detection of heart disease is crucial for timely intervention and better patient outcomes. 💓

## 🛠️ Installation

To run this project, you need to have Python installed along with the following libraries:

- `pandas` 🐼 - For data manipulation.
- `numpy` 🔢 - For numerical operations.
- `matplotlib` 📊 - For plotting graphs.
- `seaborn` 🦢 - For statistical data visualization.
- `sklearn` 📚 - For machine learning algorithms.

### Install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 Dataset

The dataset used in this project is `heart-disease.csv`, which contains patient records with the following features:

1. `age`: Age of the patient (in years) 👴👵
2. `sex`: Sex of the patient (1 = male, 0 = female) 🚹🚺
3. `cp`: Chest pain type (1-4) 💔
   - 1 = Typical angina
   - 2 = Atypical angina
   - 3 = Non-anginal pain
   - 4 = Asymptomatic
4. `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital) 💉
5. `chol`: Serum cholesterol in mg/dl 🧪
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) 🍩
7. `restecg`: Resting electrocardiographic results (0-2) 💓
   - 0 = Normal
   - 1 = ST-T wave abnormality
   - 2 = Left ventricular hypertrophy
8. `thalach`: Maximum heart rate achieved 🏃‍♂️💨
9. `exang`: Exercise-induced angina (1 = yes; 0 = no) 💪
10. `oldpeak`: ST depression induced by exercise relative to rest ⚡
11. `slope`: Slope of the peak exercise ST segment (1-3) 📉
   - 1 = Upsloping
   - 2 = Flat
   - 3 = Downsloping
12. `ca`: Number of major vessels colored by fluoroscopy (0-4) 🩸
13. `thal`: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) 🧬
14. `target`: Presence or absence of heart disease (1 = disease, 0 = no disease) ❤️

## 🏃 Steps

1. **Data Preparation** 📂:
   - Load the dataset and shuffle the rows for randomization.
   - Separate the features (`X`) and the target variable (`y`).

2. **Train-Test Split**:
   - Split the data into training and test sets (80% training, 20% testing).

3. **Model Evaluation**:
   - Evaluate multiple machine learning models, including:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - K-Nearest Neighbors Classifier
     - Gaussian Naive Bayes
     - Support Vector Classifier (SVC)

4. **Model Selection**:
   - After evaluation, the model with the highest accuracy is selected. In this case, **Gaussian Naive Bayes** was chosen as the best-performing model.

5. **Model Saving**:
   - The trained model is saved as a `.pkl` file for future predictions.

## 🤖 Model

The final model used in this project is the **Gaussian Naive Bayes** model. This model is trained on the training set and evaluated on the test set. It is then saved as a serialized `.pkl` file to make future predictions without retraining.

### Example Prediction

Once the model is trained and saved, it can be used to make predictions. For example:

```python
# Example input for prediction:
prediction = Gnb.predict(np.array([[51, 1, 2, 94, 227, 0, 1, 154, 1, 0.0, 2, 1, 3]]))
print(prediction)  # This will output the prediction for heart disease
```

## 📦 Usage

- **Train and Save the Model**: To train the model, simply run the script and it will automatically train and save the model as `model.pkl`.
- **Load and Use the Model**: After training, you can load the model for making predictions on new data.

Example code to load and use the saved model:

```python
import pickle

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions using the model
prediction = model.predict([[57, 1, 0, 150, 276, 0, 0, 112, 1, 0.6, 1, 1, 1]])
print("Prediction:", prediction)
```

## 🖼️ Screenshots

![App Screenshot]()



