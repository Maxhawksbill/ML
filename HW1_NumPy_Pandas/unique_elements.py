import numpy as np

# Create a 2D Numpy array 5*5 shape with random integers between 0 and 100
random_integers = np.random.randint(0, 100, (5, 5))
print(random_integers)

# Find the unique numbers of the array
unique_elements, counts = np.unique(random_integers, return_counts=True)
print(unique_elements)

# Find the numbers of the array that appear more than once
duplicate_numbers = unique_elements[counts>1]
print(duplicate_numbers)

# Print all the rows of the array where the sum of the row is greater than 100
rows = np.where(np.sum(random_integers, axis=1) > 300)
print(random_integers[rows])
