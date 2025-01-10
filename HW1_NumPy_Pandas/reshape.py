import numpy as np

# Create a 1D Numpy array (20,) shape with random integers between 1 and 20
random_integers = np.random.randint(1, 20, 20)
print(random_integers)

# Reshape the array to 2D 4*5 shape
reshaped_array = random_integers.reshape(4, 5)
print(reshaped_array)
