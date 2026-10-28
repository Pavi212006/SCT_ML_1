"""Demonstration of linear regression using a synthetic housing-style dataset.

Important: the generated target is artificial and is not a real house price.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Create reproducible synthetic data for a linear-regression demonstration."""
    rng = np.random.default_rng(random_state)
    rooms = rng.normal(6, 1, n_samples)
    status = rng.normal(12, 5, n_samples)
    tax = rng.normal(400, 100, n_samples)

    # Artificial formula; this is not a real-world housing-price equation.
    target = 5 * rooms - 0.5 * status - 0.01 * tax + rng.normal(0, 2, n_samples)
    return pd.DataFrame(
        {"rm": rooms, "lstat": status, "tax": tax, "synthetic_target": target}
    )


def main() -> None:
    print("Creating a synthetic housing-style dataset...")
    data = create_dataset()

    features = data[["rm", "lstat", "tax"]]
    target = data["synthetic_target"]

    # Hold out data that the model does not see during training.
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    print("\n" + "=" * 52)
    print("LINEAR REGRESSION — SYNTHETIC DATA DEMONSTRATION")
    print("=" * 52)
    print(f"Training records: {len(X_train)}")
    print(f"Test records:     {len(X_test)}")
    print(f"MAE:              {mae:.3f} synthetic target units")
    print(f"RMSE:             {rmse:.3f} synthetic target units")
    print(f"R-squared:        {r2:.3f}")
    print("\nThese scores measure performance on artificial data only.")
    print("They do not establish accuracy for real-world house prices.")

    print("\nEnter feature values to see the model output.")
    print("Use the same approximate feature scales as the generated dataset.")
    try:
        user_rooms = float(input("Average number of rooms (rm): "))
        user_status = float(input("Lower-status population percentage (lstat): "))
        user_tax = float(input("Property tax feature (tax): "))

        inputs = np.array([[user_rooms, user_status, user_tax]], dtype=float)
        if not np.isfinite(inputs).all():
            raise ValueError("All values must be finite numbers.")
        if user_rooms <= 0 or user_status < 0 or user_tax < 0:
            raise ValueError("Rooms must be positive; status and tax cannot be negative.")

        user_features = pd.DataFrame(inputs, columns=["rm", "lstat", "tax"])
        estimate = model.predict(user_features)[0]
        print(f"Predicted synthetic target: {estimate:.3f} units")
        print("This is not a price estimate in dollars or thousands of dollars.")
    except ValueError as error:
        print(f"Invalid input: {error}")

    # Evaluate only on the held-out test set.
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, test_predictions, alpha=0.7, label="Held-out test records")
    lower = min(y_test.min(), test_predictions.min())
    upper = max(y_test.max(), test_predictions.max())
    plt.plot([lower, upper], [lower, upper], "--", label="Ideal prediction")
    plt.xlabel("Actual synthetic target")
    plt.ylabel("Predicted synthetic target")
    plt.title("Linear Regression: Test-Set Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
