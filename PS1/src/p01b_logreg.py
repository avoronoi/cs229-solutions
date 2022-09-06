from distutils.log import Log
import numpy as np
import util
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # НЕ ЗАБЫТЬ РАЗОБРАТЬСЯ С ФУНКЦИЯМИ
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    
    model.fit(x_train, y_train)
    util.plot(x_train, y_train, model.theta)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval) > 0.5
    print('LogReg:', np.mean(y_pred == y_eval))
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def g(self, x):
        return 1 / (1 + np.exp(-x @ self.theta))
    
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        
        while True:
            hyp = self.g(x)
            grad_m = x.T @ (hyp - y)
            hessian_m = (hyp * (1 - hyp) * x.T) @ x
            diff = np.linalg.inv(hessian_m) @ grad_m
            self.theta -= diff
            if np.linalg.norm(diff, ord=1) < self.eps:
                break
            
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.g(x)
        # *** END CODE HERE ***
