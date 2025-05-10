import json
import sys
import pandas as pd
import numpy as np
from predict import estimate_price
from utils import normalize_mileage


def mean_squared_error(X, y, theta0, theta1):
    m = len(X)
    total_error = 0
    for i in range(m):
        error = estimate_price(X[i], theta0, theta1) - y.iloc[i]
        total_error += error ** 2
    return total_error / m

def train(data_path, output_path):
    """
    Train a simple linear regression model to predict price based on km.

    Args:
    data_path (str): The path to the CSV file containing the training data.
    output_path (str): The path to save the thetas.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # Prepare the data
    X = df['km']
    y = df['price']

    # Shuffle the data before splitting
    indices = np.arange(len(X))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    # Reorder the data
    X = X.iloc[indices]
    y = y.iloc[indices]

    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Normalize the training data
    X_train_normalized = X_train.apply(normalize_mileage)

    # Convert to numpy arrays for faster computation
    X_train_normalized = X_train_normalized.values
    y_train = y_train.values

    # Get the training set size
    m_train = len(X_train_normalized)

    # learning rate
    learning_rate = 0.01

    # Initialize the parameters
    theta0 = 0
    theta1 = 0

    for epoch in range(100_000):
        if epoch % 10_000 == 0:
            print(f"Epoch {epoch}: theta0 = {theta0}, theta1 = {theta1}")

        # Calculate errors for the entire training set
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        for i in range(m_train):
            error = estimate_price(
                X_train_normalized[i], theta0, theta1) - y_train[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * X_train_normalized[i]
        tmp_theta0 = learning_rate * (1/m_train) * sum_error_theta0
        tmp_theta1 = learning_rate * (1/m_train) * sum_error_theta1
        # Update parameters using training set size
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    # Evaluate on test set
    X_test_normalized = X_test.apply(normalize_mileage).values

    # Calculate MSE on test set
    test_mse = mean_squared_error(X_test_normalized, y_test, theta0, theta1)
    test_rmse = test_mse ** 0.5

    avg_price = sum(y) / len(y)
    price_range = max(y) - min(y)
    relative_error = test_rmse / avg_price * 100

    # Print analysis
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
    print(f"Average car price: {avg_price:.2f}")
    print(f"Price range: {price_range:.2f}")
    print(f"Relative error: {relative_error:.2f}%")
    print(
        f"Interpretation: On average, predictions are off by {test_rmse:.2f},")
    print(f"which is {relative_error:.2f}% of the average car price.")

    if relative_error < 10:
        print("This is a very good model fit for car price prediction.")
    elif relative_error < 20:
        print("This is a reasonable model fit for car price prediction.")
    else:
        print("This model has significant error. Consider additional features or non-linear models.")

    # Save the parameters and normalization values
    thetas = {'theta0': theta0, 'theta1': theta1}
    with open(output_path, 'w') as f:
        json.dump(thetas, f)

    print(f"Model parameters saved to {output_path}")

if __name__ == "__main__":
    try:
        train('data.csv', 'thetas.json')
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
