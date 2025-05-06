import sys

from utils import load_thetas, normalize_mileage


def estimate_price(km, theta0, theta1):
    """
    Simple linear regression function to predict price based on km.
    
    Args:
    km (float): The kilometers driven.
    theta0 (float): The intercept of the linear regression line.
    theta1 (float): The slope of the linear regression line.

    Returns:
    float: The predicted price.
    """
    return theta0 + theta1 * km

if __name__ == "__main__":
    try:
        mileage = input("Enter the kilometers driven: ")
        # Normalize the mileage
        mileage_normalized = normalize_mileage(float(mileage))
    except ValueError:
        print("Please provide a valid number for kilometers.")
        sys.exit(1)
    # load the theta values
    theta0, theta1 = load_thetas('thetas.json')
    predicted_price = estimate_price(mileage_normalized, theta0, theta1)
    print(f"Predicted price for {mileage} km: {predicted_price:.2f}")