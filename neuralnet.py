import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
np.random.seed(0)


# Stochastic Gradient Descent (SGD) Optimizer
class Optimizer_SGD:

    # Initialize optimizer
    # Learning rate of 1.0 is the default
    def __init__(self, learning_rate=1.0, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1.0 + self.decay * self.iterations))
                        
    # Update parameters
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = (self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates    
            bias_updates = (self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
                
        layer.weights += weight_updates
        layer.biases += bias_updates
            
    def post_update_params(self):
        self.iterations +=1
                        

class Optimizer_RMSprop:

    # Initialize optimizer
    # Learning rate of 1.0 is the default
    def __init__(self, learning_rate=0.001, decay = 0, epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1.0 + self.decay * self.iterations))
                        
    # Update parameters
    def update_params(self, layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
                
            layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights **2
            
            layer.bias_cache = self.rho * layer.bias_cache + (1-self. rho) * layer.dbiases **2
                         
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
             
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations +=1
        
        
class Optimizer_Adagrad:

    # Initialize optimizer
    # Learning rate of 1.0 is the default
    def __init__(self, learning_rate=1.0, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = 1e-7
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1.0 + self.decay * self.iterations))
                        
    # Update parameters
    def update_params(self, layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
                
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2
            
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
             
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations +=1
        
class Optimizer_Adam:
    # Initialize optimizer
    # Learning rate of 1.0 is the default
    def __init__(self, learning_rate=0.001, decay = 0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1.0 + self.decay * self.iterations))
                        
    # Update parameters
    def update_params(self, layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)
                
            layer.weight_momentums = self.beta_1 * layer.weight_momentums+ (1-self.beta_1)* layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + ( 1-self.beta_1) * layer.dbiases
            weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations+1))
            layer.weight_cache = (self.beta_2 * layer.weight_cache+ (1 - self.beta_2) * layer.dweights**2)
            layer.bias_cache = (self.beta_2 * layer.bias_cache+ (1 - self.beta_2) * layer.dbiases**2)
            weight_cache_corrected = (layer.weight_cache/ (1 - self.beta_2 ** (self.iterations + 1)))
            bias_cache_corrected = (layer.bias_cache/ (1 - self.beta_2 ** (self.iterations + 1)))
            layer.weights += (-self.current_learning_rate* weight_momentums_corrected/ (np.sqrt(weight_cache_corrected) + self.epsilon))
            layer.biases += (-self.current_learning_rate* bias_momentums_corrected/ (np.sqrt(bias_cache_corrected) + self.epsilon))
    def post_update_params(self):
        self.iterations +=1
# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
                zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix
            jacobian_matrix = (
                np.diagflat(single_output)
                - np.dot(single_output, single_output.T)
            )

            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:

    # Calculates the data loss
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        if len(y_true.shape) == 1:

            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # One-hot encoded labels
        elif len(y_true.shape) == 2:

            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # Number of labels
        labels = len(dvalues[0])

        # If labels are sparse, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Combined Softmax activation and Cross-Entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):

        # Output layer activation
        self.activation.forward(inputs)

        # Store output
        self.output = self.activation.output

        # Return loss
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second Dense layer
dense2 = Layer_Dense(64, 3)

# Create combined Softmax activation and loss
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate= 0.02,decay = 1e-5)


for epoch in range(1001):

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Calculate loss
    loss = loss_activation.forward(dense2.output, y)
    
    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    
    if not epoch %100:
        print(f' epoch: {epoch}, ' + f'acc: {accuracy:.3f}, '+ f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')
        
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()