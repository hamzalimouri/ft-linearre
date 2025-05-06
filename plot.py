import matplotlib.pyplot as plt
import csv

from predict import estimate_price
from utils import load_data, load_thetas, normalize_mileage


def plot(data_path="data.csv"):
    """
    Plot the data from the CSV file.

    Args:
    data_path (str): The path to the CSV file containing the data.
    """
    # Load the dataset
    try:
        df = load_data(data_path)
        if df.empty:
            print("The dataset is empty.")
            raise ValueError("The dataset is empty.")
    except Exception:
        print("Error loading data.")
        return
    # Prepare the data
    try:
        X = df['km']
        y = df['price']
    except KeyError:
        print("Data does not contain 'km' or 'price' columns.")
        return
    
    # Load the theta values
    theta0, theta1 = load_thetas('thetas.json')
    if theta0 is None or theta1 is None:
        print("Error loading thetas.")
        return
    # Normalize the mileage
    X_min, X_max = X.min(), X.max()
    # Calculate the predicted values
    x_line = sorted(X)
    y_line = [estimate_price(normalize_mileage(xi, X_min, X_max), theta0, theta1)
              for xi in x_line]
    
    # Plot the data
    plt.scatter(X, y, color="blue", label="Actual data")
    plt.plot(x_line, y_line, color="red", label="Predicted line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Mileage vs Price")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot()