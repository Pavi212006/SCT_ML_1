import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. GENERATE LOCAL DATA (No internet/CSV needed)
print("Initializing Local Dataset")
np.random.seed(42)
n_samples = 500
rooms = np.random.normal(6, 1, n_samples)
status = np.random.normal(12, 5, n_samples)
tax = np.random.normal(400, 100, n_samples)
# Price formula: 5*rooms - 0.5*status - 0.01*tax + noise
price = (5 * rooms) - (0.5 * status) - (0.01 * tax) + np.random.normal(0, 2, n_samples)

data = pd.DataFrame({'rm': rooms, 'lstat': status, 'tax': tax, 'medv': price})

# 2. SETUP THE MODEL
X = data[['rm', 'lstat', 'tax']]
y = data['medv']
model = LinearRegression().fit(X, y)

# 3. USER-FRIENDLY INTERFACE
print("\n" + "="*40)
print("HOUSE PRICE PREDICTOR")
print("="*40)
print("Please enter the following details to get a price estimate:")

try:
    u_rooms = float(input("\n Average Number of Rooms: "))
    u_status = float(input(" % Lower Status of Population: "))
    u_tax = float(input(" Property Tax Rate: "))

    prediction = model.predict([[u_rooms, u_status, u_tax]])
    
    print("\n" + "-"*40)
    print(f"RESULT: Estimated House Value: ${prediction[0]:,.2f}k")
    print("-"*40)
except Exception as e:
    print(f"\nError: Please enter numbers only.")

# 4. GENERATE THE CHART
print("\nOpening Performance Chart")
plt.figure(figsize=(8,5))
plt.scatter(y, model.predict(X), color='teal', alpha=0.4, label='Data Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='orange', lw=2, label='Perfect Fit')
plt.title("SkillCraft ML Task 01: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Model Predictions")
plt.legend()
plt.show()
