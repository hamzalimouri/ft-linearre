import json
import pandas as pd
from predict import estimate_price
from utils import normalize_mileage

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
    m = len(y)
    X_min, X_max = X.min(), X.max()
    X = X.apply(normalize_mileage, args=(X_min, X_max))  # Normalize the data
    X = X.values
    y = y.values
    
    # learning rate
    learning_rate = 0.001
    
    # Initialize the parameters
    theta0 = 0
    theta1 = 0

    for epoch in range(100000):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: theta0 = {theta0}, theta1 = {theta1}")
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        for i in range(m):
            error = estimate_price(X[i], theta0, theta1) - y[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * X[i]
        
        print(f"Sum Error: {sum_error_theta0}, {sum_error_theta1}")

        theta0 -= learning_rate * (1/m) * sum_error_theta0
        theta1 -= learning_rate * (1/m) * sum_error_theta1
    
    # Save the parameters
    thetas = {'theta0': theta0, 'theta1': theta1}
    with open(output_path, 'w') as f:
        json.dump(thetas, f)

if __name__ == "__main__":
    train('data.csv', 'thetas.json')
