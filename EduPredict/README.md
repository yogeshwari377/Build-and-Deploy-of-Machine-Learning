# 🎓 EduPredict – Student Academic Performance Prediction System

### Machine Learning | Flask | Python | HTML | CSS | JavaScript

EduPredict is a web-based Machine Learning application designed to predict students' academic performance based on their study habits and previous academic records.

The application provides an interactive, single-page dashboard where users can enter student details, generate a predicted Performance Index, and receive personalized academic insights.

This project was developed as part of the **Build and Deployment of Machine Learning Applications** coursework.

---

## ✨ Features

- Interactive single-page website with a light-purple and lavender UI/UX.
- Student academic performance prediction using Machine Learning.
- User-friendly input form with validation.
- Quick test presets for different student profiles.
- Predicted Performance Index visualization.
- Personalized academic guidance based on prediction results.
- Flask REST API integration.
- Responsive design for desktop and mobile devices.
- Model training and evaluation pipeline.
- Docker configuration for containerization.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| ML Models | Linear Regression, Random Forest Regressor |
| Hyperparameter Tuning | GridSearchCV |
| Model Serialization | Joblib |
| Containerization | Docker |
| Version Control | Git, GitHub |

## 📊 Dataset

**Student Performance – Multiple Linear Regression Dataset**

Source: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

The dataset contains the following attributes:

| Feature | Description |
|---|---|
| Hours Studied | Number of hours spent studying |
| Previous Scores | Previous academic performance |
| Extracurricular Activities | Participation in extracurricular activities |
| Sleep Hours | Average daily sleep duration |
| Sample Question Papers Practiced | Number of sample papers completed |
| Performance Index | Target variable to be predicted |

## 🧠 Machine Learning Workflow

1. Load the student performance dataset.
2. Perform data cleaning and preprocessing.
3. Encode categorical features.
4. Split the dataset into training and testing sets.
5. Train Linear Regression and Random Forest Regression models.
6. Perform hyperparameter tuning using GridSearchCV.
7. Evaluate model performance using MAE, RMSE and R².
8. Save the selected trained model using Joblib.
9. Load the saved model into the Flask backend.
10. Generate predictions through the web interface.

## 💻 How to Run the Project

### 1. Clone the repository

Clone the GitHub repository and navigate to the EduPredict project directory.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the dataset

Download the CSV file from the Kaggle dataset link provided above.

Place it in the following location:

```text
EduPredict/data/Student_Performance.csv
```

### 6. Train the ML model

```bash
python train_model.py
```

The training pipeline evaluates the models and saves the selected trained model.

### 7. Start the Flask application

```bash
python app.py
```

### 8. Open the website

Visit the following URL in your browser:

http://127.0.0.1:5000

**Note:** The trained model must be available before using the prediction feature.

## 🌐 Application Workflow

1. Open the EduPredict website.
2. Enter student academic and study-related details.
3. Alternatively, select a quick test preset.
4. Click **Predict Performance**.
5. The Flask backend processes the input using the trained ML model.
6. The predicted Performance Index is displayed on the same page.
7. The application presents personalized academic insights.

## 📈 Model Evaluation

The Machine Learning models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.
- **R² Score:** Measures how well the model explains variation in the target variable.

Hyperparameter tuning is performed using GridSearchCV to improve model performance.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays the EduPredict website |
| GET | `/health` | Checks application and model status |
| POST | `/predict` | Generates student performance predictions |

## 🐳 Docker Support

The project includes Docker configuration to support containerized execution.

Refer to the project's Dockerfile and configuration for build and run instructions.

## 🎯 Project Objective

The objective of EduPredict is to demonstrate the complete lifecycle of a Machine Learning application, including dataset preprocessing, model training, hyperparameter tuning, model evaluation, backend API development, frontend integration and deployment preparation.

## 📌 Disclaimer

EduPredict is an educational Machine Learning project. Its predictions are estimates based on the training dataset and should not be considered definitive assessments of a student's academic ability or future performance.

---

**Developed as a college mini-project for Build and Deployment of Machine Learning Applications.**
