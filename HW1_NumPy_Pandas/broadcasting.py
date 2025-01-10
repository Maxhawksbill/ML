import numpy as np

# Create a 2D Numpy array 3*3 shape and 1D array shape (3,) with random integers between 0 and 100
random_integers_2d = np.random.randint(0, 100, (3, 3))
random_integers_1d = np.random.randint(0, 100, (3))
print(random_integers_2d)
print(random_integers_1d)

# Add the 1D array to each row of the 2D array using broadcasting
result = random_integers_2d + random_integers_1d
print(result)
