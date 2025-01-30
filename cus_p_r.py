import numpy as np

class CustomPerceptronDotProduct:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Calculate linear combination using dot product
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply step function to get the prediction
                y_predicted = self._step_function(linear_output)

                # Update weights and bias if the prediction is incorrect
                if y[idx] != y_predicted:
                    update = self.learning_rate * (y[idx] - y_predicted)
                    # Update weights using dot product
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        # Use dot product for predictions
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_function(linear_output)

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)