import numpy as np

# Create a NumPy array from 10 random integers between 0 and 100
random_integers = np.random.randint(0, 100, 10)
print(random_integers)

# Find average of the array, median of the array, and the standard deviation of the array
average = np.mean(random_integers)
median = np.median(random_integers)
standard_deviation = np.std(random_integers).round(3)
print("Average: ", average)
print("Median: ", median)
print("Standard Deviation: ", standard_deviation)

# Replace all the even numbers in the array with 0
# One way to do this is to loop through the array and replace the even numbers with 0
for i in range(len(random_integers)):
    if random_integers[i] % 2 == 0:
        random_integers[i] = 0
print(random_integers)

# Another way to do this is to use list comprehension
random_integers = np.array([0 if int(x) % 2 == 0 else int(x) for x in random_integers])
print(random_integers)

# Another way to do this is to use NumPy vectorization
random_integers[random_integers % 2 == 0] = 0
print(random_integers)
