import numpy as np
class NeuralNetwork:
    def __init__(self):      
        self.input_size = 10
        self.output_size = 1
        self.hidden_size = 5
        self.learning_rate = 0.1
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W_hidden = []
        self.b_hidden = []
        for _ in range(4):  
            self.W_hidden.append(np.random.randn(self.hidden_size, self.hidden_size))
            self.b_hidden.append(np.zeros((1, self.hidden_size)))

        # weights and biases for the output layer
        self.W_output = np.random.randn(self.hidden_size, self.output_size)
        self.b_output = np.zeros((1, self.output_size))

    def forward_propagation(self, X):
        # Forward propagation for the first two input nodes
        hidden_layer1 = np.dot(X[:, :2], self.W1[:2, :]) + self.b1
        hidden_layer1 = self.sigmoid(hidden_layer1)

        # Forward propagation for the remaining input nodes
        hidden_layers = [hidden_layer1]
        for i in range(4):
            hidden_layer = np.dot(hidden_layers[i], self.W_hidden[i]) + self.b_hidden[i]
            hidden_layer = self.sigmoid(hidden_layer)
            hidden_layers.append(hidden_layer)

        # Forward propagation for the output layer
        output_layer = np.dot(hidden_layers[-1], self.W_output) + self.b_output
        predicted_output = self.sigmoid(output_layer)
        return predicted_output, hidden_layers

    def backward_propagation(self, X, y, predicted_output, hidden_layers):
        # gradients for the output layer
        output_error = (predicted_output - y) * self.sigmoid_derivative(predicted_output)
        dW_output = np.dot(hidden_layers[-1].T, output_error)
        db_output = np.sum(output_error, axis=0)

        # gradients for the hidden layers
        dW_hidden = []
        db_hidden = []
        error = np.dot(output_error, self.W_output.T)
        for i in range(3, -1, -1):
            hidden_error = np.dot(error, self.W_hidden[i].T) * self.sigmoid_derivative(hidden_layers[i])
            dW_hidden.insert(0, np.dot(hidden_layers[i - 1].T, hidden_error))
            db_hidden.insert(0, np.sum(hidden_error, axis=0))
            error = hidden_error

        #gradients for the first hidden layer
        dW1 = np.dot(X[:, :2].T, error[:, :2])  # Only consider the first two input nodes
        db1 = np.sum(error[:, :2], axis=0, keepdims=True)

        self.W_output -= self.learning_rate * dW_output
        self.b_output -= self.learning_rate * db_output
        for i in range(4):
            self.W_hidden[i] -= self.learning_rate * dW_hidden[i]
            self.b_hidden[i] -= self.learning_rate * db_hidden[i]
       
        self.W1[:2, :2] -= self.learning_rate * dW1[:2, :]
        self.b1[:, :2] -= self.learning_rate * db1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predicted_output, hidden_layers = self.forward_propagation(X)

            self.backward_propagation(X, y, predicted_output, hidden_layers)

            loss = np.mean(np.square(predicted_output - y))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.8f}")
    
X = np.random.rand(10, 10000)
y = np.random.randint(0, 2, (10, 1))

network = NeuralNetwork()
network.train(X, y, epochs=100)
