import numpy as np

# Create example arrays
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])  # Shape: (4, 4)

B = np.array([[0],
              [1],
              [2],
              [3]])  # Shape: (4, 1)

# Reshape B to (4, 1, 1) for broadcasting
B_reshaped = B[:, np.newaxis, :]  # Shape becomes (4, 1, 1)

# Perform the subtraction (resulting shape will be (4, 4, 4))
result = A - B_reshaped

# Show the result of A - B_reshaped
print("A - B_reshaped = \n", result)
print("Shape of result: ", result.shape)
final_result = np.sum(result, axis=0)

# Show the summed result
print("Sum along axis 0: \n", final_result)
print("Shape of final result: ", final_result.shape)
