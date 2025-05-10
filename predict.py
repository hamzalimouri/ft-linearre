import sys

from utils import load_params, normalize_mileage


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
    except Exception as e:
        print(f"An error occurred while normalizing mileage: {e}")
        sys.exit(1)
    # load the theta values
    try:
        params = load_params('thetas.json')
        theta0, theta1 = params['theta0'], params['theta1']
    except FileNotFoundError:
        print("File thetas.json not found.")
        sys.exit(1)
    if theta0 is None or theta1 is None:
        print("Error loading thetas.")
        sys.exit(1)
    try:
        predicted_price = estimate_price(mileage_normalized, theta0, theta1)
        print(f"Predicted price for {mileage} km: {predicted_price:.2f}")
    except Exception as e:
        print(f"An error occurred while predicting the price: {e}")
        sys.exit(1)