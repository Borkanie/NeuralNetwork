import numpy as np

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# Computing the dot product of input_vector and weights_1
dot_product_1 = np.dot(input_vector,weights_1)
print(f"The first dot product is:{dot_product_1}")

# Computing the dot product of input_vector and weights_2
dot_product_2 = np.dot(input_vector,weights_2)
print(f"The first dot product is:{dot_product_2}")