import numpy as np


class RVR():
    def __init__(self):
        """ """
        alpha_bound = 10e12
        weight_bound = 10e-6
        
        
        self.weights = None
        self.sigma = None
        self.d = None
        self.alpha = None
        self.beta = None
        
        
    def update_w_sigma(self, data, target):
        """Calculate the mean and the covariance matrix
           of the posterior distribution"""
    
        n, d = data.shape

        self.alpha_bound = 10e12
        self.weight_bound = 10e-6

        self.sigma = np.zeros((d, d))
        self.weights = np.zeros(d)

        vector_mask_a = self.alpha < self.alpha_bound
        matrix_mask_a = np.ix_(vector_mask_a, vector_mask_a)


        self.sigma[matrix_mask_a] = np.linalg.inv(self.beta * np.dot(data[:, vector_mask_a].T, \
                                                    data[:, vector_mask_a]) + np.diag(self.alpha[vector_mask_a]))

        self.weights[vector_mask_a] = self.beta * np.dot(self.sigma[matrix_mask_a],
                                                         np.dot(data[:, vector_mask_a].T, target))



        vector_mask_w = np.abs(self.weights) < self.weight_bound
        matrix_mask_w = np.ix_(vector_mask_w, vector_mask_w)

        self.weights[vector_mask_w] = 0
        self.sigma[matrix_mask_w] = 0
    
        return None
    
    
    def update_alpha_beta(self, data, target):
        """Update the hyperparameters to increase evidence"""


        n, d = data.shape

        self.update_w_sigma(data, target)

        alpha_new = self.alpha
        beta_new = self.beta

        mask = ((self.alpha < self.alpha_bound) & (np.abs(self.weights) > self.weight_bound))

        cache = 1 - self.alpha[mask] * np.diag(self.sigma)[mask]

        alpha_new[mask] = cache / (self.weights[mask] ** 2)
        alpha_new[~mask] = np.inf
        alpha_new[alpha_new > self.alpha_bound] = np.inf

        beta_new = (n - cache.sum()) / (np.linalg.norm(target - np.dot(data, self.weights)) ** 2)
        
        
        self.alpha = alpha_new
        self.beta = beta_new
        return None
        
    def fit(self, data, target, max_iter=1000, is_tqdm=False):
        
        """Train the Relevance Vector Regression model"""
        n, d = data.shape
        
        self.d = d
        
        
        if n != target.shape[0]:
            raise Exception("N_objects in data and in target are not the same")
        
        self.alpha = np.ones((self.d))
        self.beta = 1
        
        if is_tqdm:
            from tqdm.notebook import tqdm
            for _ in tqdm(range(max_iter)):
                self.update_alpha_beta(data, target)
        else:
            for _ in range(max_iter):
                self.update_alpha_beta(data, target)
                
                
        self.update_w_sigma(data, target)

        return None
    
    def predict(self, data):
        """Predicts target using trained model"""
        if self.d != data.shape[1]:
            raise Exception("Your data is not the same size as trained")
        
        return data.dot(self.weights.ravel())