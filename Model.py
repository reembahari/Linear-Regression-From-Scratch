import random

# Generate training data
x_data = [i for i in range(10)]  # Input values
y_data = [2 * x + 1 + random.uniform(-1, 1) for x in x_data]  # Output values with random noise

# Initialize parameters
m, b = 0, 0  # Initial slope and intercept
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Predict y values based on current m and b
def predict(x, m, b):
    return [m * xi + b for xi in x]

# Calculate mean squared error
def compute_loss(y_pred, y_actual):
    return sum((y_pred[i] - y_actual[i]) ** 2 for i in range(len(y_actual))) / len(y_actual)

# Update m and b using gradient descent
def update_weights(x, y, m, b, alpha):
    n = len(y)
    m_grad = -(2 / n) * sum(x[i] * (y[i] - (m * x[i] + b)) for i in range(n))
    b_grad = -(2 / n) * sum(y[i] - (m * x[i] + b) for i in range(n))
    return m - alpha * m_grad, b - alpha * b_grad

# Train the model
for i in range(iterations):
    y_pred = predict(x_data, m, b)  # Predict y values
    m, b = update_weights(x_data, y_data, m, b, alpha)  # Update parameters
    if i % 100 == 0:  # Print loss every 100 iterations
        print(f"Iteration {i}, Loss: {compute_loss(y_pred, y_data):.4f}")

# Final results
print(f"Trained model: m = {m:.2f}, b = {b:.2f}")
print(f"Predictions: {predict(x_data, m, b)}")
