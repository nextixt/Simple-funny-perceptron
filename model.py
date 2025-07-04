import random

def neuron(inputs, weights):
    sum_ = sum(x * w for x, w in zip(inputs, weights))
    return 1 if sum_ >= 0 else 0

# Learning parameters
INPUT_SIZE = 5          # Input_size
LEARNING_RATE = 0.01    # Learning_rate
EPOCHS = 1000           # Epochs

# Random weights
weights = [random.uniform(-1, 1) for _ in range(INPUT_SIZE)]

print("Start learning")

for epoch in range(EPOCHS):
    # 1. Random input from -10 to 10
    inputs = [random.uniform(-10, 10) for _ in range(INPUT_SIZE)]
    
    # 2. Right answer (if sum of inputs > 0 → 1, else 0)
    correct_answer = 1 if sum(inputs) > 0 else 0
    
    # 3. Prediction from NN
    prediction = neuron(inputs, weights)
    
    # 4. If error, correct weights
    if prediction != correct_answer:
        for i in range(INPUT_SIZE):
            if inputs[i] > 0:
                weights[i] += LEARNING_RATE * (correct_answer - prediction)
            else:
                weights[i] -= LEARNING_RATE * (correct_answer - prediction)
    
    # Show progress every 100 epoch
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Weights = {[round(w, 2) for w in weights]}")

# Test
test_inputs = [5, -3, 2, -1, 4]  # Sum = 7 → correct answer: 1
print(f"\n Result after learning: {neuron(test_inputs, weights)} (Must be 1)")
