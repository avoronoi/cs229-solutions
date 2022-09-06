import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    
    model.fit(x_train, y_train)
    util.plot(x_train, y_train, model.theta)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval) > 0.5
    print('GDA:', np.mean(y_pred == y_eval))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = np.mean(y)
        mu = {}
        mu[0] = np.mean([x[i] for i in range(m) if y[i] == 0], 
                        axis=0)
        mu[1] = np.mean([x[i] for i in range(m) if y[i] == 1], 
                        axis=0)
        sigma = np.mean([(np.matrix(x[i] - mu[y[i]]).T
                        @ np.matrix(x[i] - mu[y[i]]))
                        for i in range(m)], axis=0)
        sigma_inv = np.linalg.inv(sigma)
        
        self.theta = np.zeros(n + 1)
        self.theta[1:] = sigma_inv @ (mu[1] - mu[0])
        self.theta[0] = (1/2 * (mu[0] + mu[1]) @ sigma_inv @ (mu[0] - mu[1]) 
                         - np.log((1 - phi) / phi))
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
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE
