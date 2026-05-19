# 🎓 Student Performance Prediction System

## Description
This project predicts student academic performance using Machine Learning (Linear Regression). It analyzes study behavior and academic history to forecast final examination scores.

## 📁 Project Structure
```
Student_Performance_Prediction_System/
│
├── dataset/
│   └── student_performance_dataset.csv    # Dataset with 5000 student records
│
├── notebook/
│   └── student_performance_prediction.ipynb  # Jupyter Notebook with full analysis
│
├── source_code/
│   └── main.py                            # Standalone Python script
│
├── screenshots/
│   ├── dataset_output.png
│   ├── heatmap.png
│   ├── scatterplot.png
│   └── prediction_result.png
│
├── documentation/
│   └── Project_Report.pdf                 # Detailed project report
│
├── README.md
└── requirements.txt
```

## 🛠 Technologies Used
- **Python** – Programming Language
- **Pandas** – Data Handling & Manipulation
- **NumPy** – Numerical Operations
- **Matplotlib** – Data Visualization
- **Seaborn** – Advanced Statistical Visualization
- **Scikit-learn** – Machine Learning (Linear Regression)

## ✨ Features
- 📊 Data Cleaning & Preprocessing
- 📈 Data Visualization (Scatter Plot, Heatmap, Histogram, Pair Plot)
- 🤖 Model Training using Linear Regression
- 🎯 Student Score Prediction
- 📉 Performance Evaluation (MAE, MSE, R² Score)

## 📊 Dataset
The dataset contains **5000 student records** with the following attributes:

| Attribute | Description |
|-----------|-------------|
| StudyHours | Number of study hours per day |
| Attendance | Student attendance percentage |
| SleepHours | Average sleep hours per day |
| PreviousScore | Previous examination score |
| FinalScore | Final predicted score (Target) |

## 🧠 Algorithm Used
**Linear Regression** – A supervised machine learning algorithm for predicting continuous numerical values.

## 🚀 How to Run

### Option 1: Using Jupyter Notebook
```bash
pip install -r requirements.txt
jupyter notebook notebook/student_performance_prediction.ipynb
```

### Option 2: Using Python Script
```bash
pip install -r requirements.txt
python source_code/main.py
```

### Option 3: Google Colab
Upload the notebook and dataset to Google Colab and run all cells.

## 📈 Performance Metrics
- **Mean Absolute Error (MAE)** – Average prediction error
- **Mean Squared Error (MSE)** – Squared prediction error
- **R² Score** – Model accuracy and goodness of fit

## 👤 Author
Student Name

## 📄 License
This project is for educational purposes.
