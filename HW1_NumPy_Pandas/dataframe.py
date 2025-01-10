import pandas as pd

# Create a DataFrame with 5 rows and 3 columns with random name, age, and city
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}
df = pd.DataFrame(data)
print(df)

# Add a new column to the DataFrame Salary with random integers between 50,000 and 100,000
df['salary'] = pd.Series([50000, 60000, 70000, 80000, 90000])
print(df)

# Filter the DataFrame to show only the rows where the age is greater than 30 and the salary is greater than 70,000
filtered_df = df[(df['age'] > 30) & (df['salary'] > 70000)]
print(filtered_df)
