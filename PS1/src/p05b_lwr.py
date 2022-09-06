import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    pred = model.predict(x_valid)
    m = x_valid.shape[0]
    print(1 / m * np.linalg.norm(pred - y_valid, 2)**2)
    
    # Plot validation predictions on top of training set
    plt.scatter(x_valid[:, 1], pred, marker='o')
    plt.scatter(x_train[:, 1], y_train, marker='x')
    plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        
        self.x = x
        self.y = y
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        result = np.zeros(x.shape[0])
        for i, f in enumerate(x):        
            w = np.exp(-np.linalg.norm(f - self.x, 2, axis=1)**2 / (2 * self.tau * self.tau))
            theta = (np.linalg.inv((self.x.T * w) @ self.x) 
                     @ (self.x.T @ (w * self.y)))
            result[i] = f @ theta
        return result
        # *** END CODE HERE ***
