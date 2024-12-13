import streamlit as st

# Linear regression functions
def predict(x, m, b):
    return [m * xi + b for xi in x]

# User interface
st.title("Linear Regression From Scratch")
st.write("Enter the values to make predictions using the model")

# User inputs
m = st.number_input("Enter the slope (m):", value=0.0)
b = st.number_input("Enter the intercept (b):", value=0.0)
x_input = st.text_input("Enter the values of x (comma-separated):", "1, 2, 3, 4, 5")

if st.button("Predict"):
    try:
        x_values = [float(i) for i in x_input.split(",")]
        predictions = predict(x_values, m, b)
        st.write(f"Input values: {x_values}")
        st.write(f"Predictions: {predictions}")
    except ValueError:
        st.error("Please ensure the values are entered correctly!")
