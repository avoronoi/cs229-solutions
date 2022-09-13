from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def find_assignment(X, centroids):
    assignment = np.zeros(X.shape[0], dtype=np.int32)
    for i in range(len(assignment)):
        assignment[i] = np.argmin(np.linalg.norm(X[i] - centroids, axis=1))
    return assignment

def k_means(X, k, trial_num=1, max_iter=np.inf):    
    best_cost = np.inf
    best_centroids = None
    
    for trial in range(trial_num):
        centroids = np.float64(X[np.random.choice(X.shape[0], size=k, replace=False)])
        it = 0
        old_cost, cost = None, None
        
        while it < max_iter and (old_cost is None or old_cost != cost):
            assignment = find_assignment(X, centroids)
            
            # Update centroids
            new_centroids = centroids.copy()
            cluster_sizes = np.bincount(assignment)
            new_centroids[cluster_sizes != 0] = 0
            for i, centroid in enumerate(assignment):
                new_centroids[centroid] += X[i]
            new_centroids[cluster_sizes != 0] /= cluster_sizes[cluster_sizes != 0, np.newaxis]
            
            # Calculate new cost
            old_cost = cost
            cost = 0
            for i, centroid in enumerate(assignment):
                try:
                    cost += np.linalg.norm(X[i] - new_centroids[centroid])
                except RuntimeWarning:
                    print(new_centroids)
                    exit()
            it += 1
            
        # Choose the best set of centroids
        if cost < best_cost:
            best_cost = cost
            best_centroids = centroids

    return best_centroids

CLUSTER_NUM = 16

if __name__ == '__main__':
    np.random.seed(229)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    
    X = np.float64(imread('../data/peppers-small.tiff')) / 255
    X.shape = (X.shape[0] * X.shape[1], X.shape[2])
    centroids = k_means(X, CLUSTER_NUM, trial_num=5)
    
    assignment = find_assignment(X, centroids)
    for i in range(CLUSTER_NUM):
        indices = assignment == i
        ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], 
                   c=np.expand_dims(centroids[i], axis=0))
    plt.show()
        
    large = np.float64(imread('../data/peppers-large.tiff')) / 255
    large_shape = large.shape
    large.shape = (large.shape[0] * large.shape[1], large.shape[2])
    large_assignment = find_assignment(large, centroids)
    large_compressed = centroids[large_assignment]
    plt.imshow(large_compressed.reshape(large_shape))
    plt.show()