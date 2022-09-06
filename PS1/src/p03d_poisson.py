import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    pred = model.predict(x_eval)
    print(np.round(np.abs((y_eval - pred) / y_eval), 4))
    
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        while True:
            hyp = np.exp(x @ self.theta)
            diff = self.step_size / m * x.T @ (hyp - y)
            self.theta -= diff
            if np.linalg.norm(diff, ord=1) < self.eps:
                break
            
        return self.theta
    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        
        return np.exp(x @ self.theta)
        
        # *** END CODE HERE ***
