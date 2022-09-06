import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    m = x_valid.shape[0]
    
    best_tau = None
    best_mse = np.inf
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        mse = 1 / m * np.linalg.norm(pred - y_valid, 2)**2
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    pred = model.predict(x_test)
    print(1 / m * np.linalg.norm(pred - y_test, 2)**2)
    
    plt.scatter(x_test[:, 1], pred, marker='o')
    plt.scatter(x_train[:, 1], y_train, marker='x')
    plt.show()
    
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
