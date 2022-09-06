import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    
    train_x, train_t = util.load_dataset(train_path, label_col='t', 
                                         add_intercept=True)
    
    model = LogisticRegression()
    model.fit(train_x, train_t)
    
    test_x, test_t = util.load_dataset(test_path, label_col='t', 
                                       add_intercept=True)
    pred_t = model.predict(test_x) > 0.5
    print(np.mean(pred_t == test_t))
    util.plot(test_x, test_t, model.theta)
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    
    train_x, train_y = util.load_dataset(train_path, label_col='y', 
                                         add_intercept=True)
    
    model = LogisticRegression()
    model.fit(train_x, train_y)
    
    test_x, test_t = util.load_dataset(test_path, label_col='t', 
                                       add_intercept=True)
    pred_t = model.predict(test_x) > 0.5
    print(np.mean(pred_t == test_t))
    util.plot(test_x, test_t, model.theta)
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    train_x, train_y = util.load_dataset(train_path, label_col='y', 
                                         add_intercept=True)
    
    model = LogisticRegression()
    model.fit(train_x, train_y)
    
    valid_x, valid_y = util.load_dataset(valid_path, label_col='y', 
                                         add_intercept=True)
    scale = np.mean(model.predict(valid_x[valid_y == 1]))
    
    test_x, test_t = util.load_dataset(test_path, label_col='t', 
                                       add_intercept=True)
    pred_t = model.predict(test_x) / scale > 0.5
    print(np.mean(pred_t == test_t))
    new_theta = model.theta.copy()
    print(model.theta)
    new_theta[0] += np.log(2 / scale - 1)
    print(new_theta)
    util.plot(test_x, test_t, new_theta)
    
    # *** END CODER HERE
