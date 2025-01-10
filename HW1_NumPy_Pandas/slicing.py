import numpy as np

# Create 2D Numpy array 3*3 shape with random integers between 0 and 100
random_integers = np.random.randint(0, 100, (3, 3))
print(random_integers)

# Print the first row of the array
print(random_integers[0])

# Print the last column of the array
print(random_integers[:, -1])

# Print the diagonal of the array
print(np.diag(random_integers))
