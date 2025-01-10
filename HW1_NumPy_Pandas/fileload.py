import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:/Users/delem/Downloads/wine.csv')

# Add a header where first column name category and the last is price
df.columns = ['category', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
              'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
              'od280/od315_of_diluted_wines', 'price']

# Add an unique index to the DataFrame
df.index = range(1, len(df) + 1)

# Print the first 5 rows of the DataFrame
print(df.head())

# Calculate the common statistics of the DataFrame
print(df.describe())

# Find the unique categories of the DataFrame
unique_categories = df['category'].unique()
print(unique_categories)
