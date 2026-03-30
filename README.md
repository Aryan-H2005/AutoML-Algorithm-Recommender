# 🤖 AutoML Algorithm Recommender System

## 🚀Live Demo : https://automl-algorithm-recommender.streamlit.app/

## 📌 Overview

The **Algorithm Recommender System** is a Data Science application that automatically analyzes a given dataset and suggests suitable machine learning algorithms. It detects whether the problem is **classification or regression**, trains multiple models, and identifies the best-performing algorithm.

---

## 🎯 Objectives

* Automate selection of machine learning algorithms
* Reduce manual effort in model selection
* Provide performance comparison of models
* Help beginners understand ML workflows

---

## ⚙️ Features

* 📂 Upload any CSV dataset
* 🔍 Automatic dataset analysis (EDA)
* 🧠 Detects problem type (Classification / Regression)
* ⚙️ Runs multiple ML algorithms
* 📊 Displays model performance
* 🏆 Suggests best algorithm

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** Streamlit

---

## 📁 Project Structure

```
MINI PROJECT/
│── app.py
│── requirements.txt
│── source/
│   │── __init__.py
│   │── pipeline.py
```

---

## ▶️ How to Run

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

### Step 2: Run application

```
streamlit run app.py
```

### Step 3: Open browser

```
http://localhost:8501
```

---

## 📊 Algorithms Used

### Classification

* Logistic Regression
* Decision Tree
* Random Forest

### Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

## 🧠 Working Process

1. Upload dataset
2. Select target column
3. Data preprocessing (handling missing values, encoding)
4. Problem type detection
5. Model training
6. Performance evaluation
7. Best model selection

---

## ⚠️ Limitations

* Basic rule-based problem detection
* No hyperparameter tuning
* No feature engineering
* Works best on clean datasets

---

## 🚀 Future Enhancements

* Add GridSearchCV for tuning
* Auto feature scaling
* Add clustering support (no target datasets)
* Deploy on cloud (Render / Streamlit Cloud)
* Improve UI/UX

---

## 📌 Conclusion

This project demonstrates how machine learning workflows can be automated to assist in selecting appropriate algorithms. It provides a practical understanding of model comparison and evaluation.

---

## 👨‍💻 Author

Aryan Harke
Third Year Computer Science Student

---

## 📄 License

This project is for educational purposes.
