# House Price Prediction - Machine Learning Task 01
### Internship: SkillCraft Technology (ML Track)

## 📌 Project Overview
This project implements a **Linear Regression** model to predict housing prices. It features a user-friendly command-line interface (CLI) that allows users to input specific parameters—such as the number of rooms and tax rates—to receive an instantaneous price estimation.

## 🚀 Key Features
- **Interactive CLI:** A "User-Friendly" prompt system for real-time predictions.
- **Data Visualization:** Includes an 'Actual vs. Predicted' scatter plot to visualize model accuracy.
- **Robust Logic:** Built using the `Scikit-Learn` library for professional-grade linear modeling.
- **Error Handling:** Validates user inputs to ensure the program doesn't crash on invalid data.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Libraries:** 
  - `Pandas`: For data manipulation.
  - `Scikit-Learn`: For the Linear Regression algorithm.
  - `Matplotlib`: For generating performance charts.
  - `NumPy`: For numerical operations and data generation.

## 📊 Model Performance
The model uses a synthesized dataset based on standard housing features. It maps the relationship between:
1. **Rooms (RM):** Average number of rooms per dwelling.
2. **Status (LSTAT):** Percentage of the lower status of the population.
3. **Tax:** Full-value property-tax rate.

## 📖 How to Run
1. Ensure you have the required libraries:
   pip install pandas scikit-learn matplotlib numpy
2. python SCT_01.py
