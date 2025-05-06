import sys

from predict import estimate_price
from utils import load_data, load_thetas, normalize_mileage

def mean_squared_error(X, y, theta0, theta1):
    m = len(X)
    total_error = 0
    for i in range(m):
        error = estimate_price(X[i], theta0, theta1) - y[i]
        print(f"Error for {i}: {error}")
        total_error += error ** 2
    return total_error / m

if __name__ == "__main__":
    print("Calculating Mean Squared Error...")
    # Load the dataset
    try:
        df = load_data('data.csv')
    except Exception:
        print("Error loading data.")
        raise
    # Prepare the data
    try:
        X = df['km']
        y = df['price']
    except KeyError:
        print("Data does not contain 'km' or 'price' columns.")
        raise
    # Normalize the mileage
    X_min, X_max = X.min(), X.max()
    X = X.apply(normalize_mileage, args=(X_min, X_max))  # Normalize the data
    X, y = X.values, y.values
    
    # Load the theta values
    try:
        theta0, theta1 = load_thetas('thetas.json')
    except FileNotFoundError:
        print("File thetas.json not found.")
        sys.exit(1)

    # Calculate the mean squared error
    mse = mean_squared_error(X, y, theta0, theta1)
    rmse = mse ** 0.5
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
