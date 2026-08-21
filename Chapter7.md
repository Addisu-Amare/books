# Chapter 7: The Scientific Python Stack

## Chapter Overview

Python's power for AI and data science comes from its rich ecosystem of libraries. This chapter focuses on the core scientific Python stack: NumPy, Pandas, and Matplotlib. NumPy provides the foundation for numerical computing with its powerful array objects. Pandas builds on NumPy to offer high-level data manipulation and analysis tools. Matplotlib enables data visualization—essential for exploratory data analysis and communicating results. We'll also cover working with APIs, testing, and debugging techniques that are crucial for building robust AI applications.

**Learning Objectives:**
By the end of this chapter, you will be able to:
- Use NumPy for efficient numerical computations with arrays.
- Manipulate and analyze data using Pandas DataFrames.
- Create visualizations with Matplotlib to explore and present data.
- Fetch data from web APIs and handle common data formats.
- Write tests for your code and debug effectively.

---

## 7.1 NumPy: The Foundation of Numerical Computing

NumPy (Numerical Python) is the fundamental library for scientific computing in Python. It provides:
- A powerful N-dimensional array object (`ndarray`).
- Fast vectorized operations (no Python loops).
- Broadcasting capabilities.
- Linear algebra, Fourier transform, and random number generation.

### 7.1.1 Installation and Import

```bash
pip install numpy
```

```python
import numpy as np

# Check version
print(np.__version__)
```

### 7.1.2 The NumPy Array (ndarray)

The `ndarray` is a homogeneous, multidimensional array of fixed size.

**Creating Arrays:**

```python
# From a list
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]

# 2D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# Special arrays
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
identity = np.eye(4)            # 4x4 identity matrix
empty = np.empty((2, 2))        # Uninitialized (garbage values)

# Ranges
range_arr = np.arange(0, 10, 2)  # [0 2 4 6 8]
linspace = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]

# Random arrays
random_uniform = np.random.rand(3, 4)     # Uniform [0, 1)
random_normal = np.random.randn(3, 4)     # Standard normal
random_int = np.random.randint(0, 10, size=(2, 3))  # Integers

print("zeros:\n", zeros)
print("\nrandom_normal:\n", random_normal)
```

**Array Attributes:**

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)      # (2, 3) - dimensions
print(arr.size)       # 6 - total elements
print(arr.ndim)       # 2 - number of dimensions
print(arr.dtype)      # int64 - data type
print(arr.itemsize)   # 8 - bytes per element
print(arr.nbytes)     # 48 - total bytes
```

**Data Types:**

```python
# Explicit dtype specification
arr_int = np.array([1, 2, 3], dtype=np.int64)
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_bool = np.array([True, False, True])

# Type conversion
arr_float = arr_int.astype(np.float32)
```

### 7.1.3 Array Operations

**Element-wise Operations:**

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Arithmetic (element-wise)
print(a + b)    # [ 6  8 10 12]
print(a - b)    # [-4 -4 -4 -4]
print(a * b)    # [ 5 12 21 32]
print(a / b)    # [0.2 0.333 0.429 0.5]
print(a ** 2)   # [ 1  4  9 16]

# Comparison (element-wise)
print(a > 2)    # [False False  True  True]
print(a == b)   # [False False False False]

# Math functions
print(np.sqrt(a))        # [1. 1.414 1.732 2.]
print(np.exp(a))         # [2.718 7.389 20.085 54.598]
print(np.log(a))         # [0. 0.693 1.099 1.386]
print(np.sin(a))         # [0.841 0.909 0.141 -0.757]
print(np.floor([1.5, 2.9]))  # [1. 2.]
print(np.ceil([1.5, 2.9]))   # [2. 3.]
print(np.round([1.5, 2.9]))  # [2. 3.]
```

**Aggregation Functions:**

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.sum(arr))           # 45 (sum all)
print(np.sum(arr, axis=0))   # [12 15 18] (sum columns)
print(np.sum(arr, axis=1))   # [ 6 15 24] (sum rows)

print(np.mean(arr))          # 5.0
print(np.median(arr))        # 5.0
print(np.std(arr))           # 2.581...
print(np.var(arr))           # 6.666...
print(np.min(arr))           # 1
print(np.max(arr))           # 9
print(np.argmax(arr))        # 8 (index of max value)
print(np.argmin(arr))        # 0 (index of min value)

# Cumulative operations
print(np.cumsum(arr))        # [ 1  3  6 10 15 21 28 36 45]
print(np.cumprod(arr))       # [1 2 6 24 120 720 5040 40320 362880]
```

### 7.1.4 Indexing, Slicing, and Iteration

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Basic indexing
print(arr[0, 0])    # 1
print(arr[1, 2])    # 7
print(arr[-1, -1])  # 12

# Slicing (rows, columns)
print(arr[0:2, 1:3])  # [[2 3]
                       #  [6 7]]

print(arr[:, 1:3])    # [[ 2  3]
                       #  [ 6  7]
                       #  [10 11]]

print(arr[1:, :2])    # [[5 6]
                       #  [9 10]]

# Fancy indexing (using arrays of indices)
indices = np.array([0, 2])
print(arr[indices])   # [[1 2 3 4]
                       #  [9 10 11 12]]

# Boolean indexing
bool_mask = arr > 5
print(arr[bool_mask])  # [ 6  7  8  9 10 11 12]

# More concise
print(arr[arr > 5])    # [ 6  7  8  9 10 11 12]

# Combining conditions
print(arr[(arr > 5) & (arr < 10)])  # [6 7 8 9]

# where function
print(np.where(arr > 5, arr, 0))  # Replace <=5 with 0
# [[0 0 0 0]
#  [0 6 7 8]
#  [9 10 11 12]]
```

### 7.1.5 Array Manipulation

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Reshaping
print(arr.reshape(3, 2))
# [[1 2]
#  [3 4]
#  [5 6]]

print(arr.flatten())  # [1 2 3 4 5 6]
print(arr.ravel())    # Same as flatten (but returns view if possible)

# Transpose
print(arr.T)
# [[1 4]
#  [2 5]
#  [3 6]]

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.vstack((a, b)))   # [[1 2 3]
                            #  [4 5 6]]
print(np.hstack((a, b)))   # [1 2 3 4 5 6]

# Splitting
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
split_arr = np.split(arr, 2, axis=0)  # Split into 2 arrays along rows
print(split_arr[0])  # [[1 2 3]
                      #  [4 5 6]]
print(split_arr[1])  # [[7 8 9]
                      #  [10 11 12]]
```

### 7.1.6 Broadcasting

Broadcasting allows operations between arrays of different shapes.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

print(a + b)  # b is broadcast to match shape of a
# [[11 22 33]
#  [14 25 36]]

# Scalar broadcast
print(a * 2)
# [[ 2  4  6]
#  [ 8 10 12]]

# Broadcasting rules:
# 1. If arrays have different dimensions, prepend 1s to smaller shape
# 2. If shapes don't match in any dimension, stretch dimension with size 1
# 3. If still don't match, raise error
```

### 7.1.7 Linear Algebra with NumPy

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print(np.dot(a, b))  # [[19 22]
                      #  [43 50]]
print(a @ b)         # Same (Python 3.5+)

# Other linear algebra operations
print(np.linalg.inv(a))       # Inverse
print(np.linalg.det(a))       # Determinant
print(np.linalg.eig(a))       # Eigenvalues and eigenvectors
print(np.linalg.solve(a, [1, 2]))  # Solve Ax = b
print(np.linalg.norm(a))      # Norm
```

### 7.1.8 Performance Comparison

NumPy is much faster than pure Python loops:

```python
import time

# Pure Python
size = 1000000
python_list = list(range(size))

start = time.time()
python_sum = sum(python_list)
python_time = time.time() - start

# NumPy
np_array = np.arange(size)

start = time.time()
np_sum = np.sum(np_array)
np_time = time.time() - start

print(f"Python sum: {python_sum:.2f}, Time: {python_time:.4f}s")
print(f"NumPy sum: {np_sum:.2f}, Time: {np_time:.4f}s")
print(f"NumPy is {python_time/np_time:.1f}x faster")
```

---

## 7.2 Pandas: Data Manipulation and Analysis

Pandas is built on NumPy and provides high-level data structures: Series (1D) and DataFrame (2D). It's essential for data cleaning, transformation, and analysis.

### 7.2.1 Installation and Import

```bash
pip install pandas
```

```python
import pandas as pd
import numpy as np
```

### 7.2.2 Series

A Series is a 1D labeled array.

```python
# Creating Series
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# With custom index
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2)
# a    10
# b    20
# c    30
# dtype: int64

# From dictionary
s3 = pd.Series({'a': 100, 'b': 200, 'c': 300})
print(s3)
# a    100
# b    200
# c    300
# dtype: int64

# Accessing
print(s2['a'])      # 10
print(s2[['a', 'c']])  # a    10, c    30

# Operations
print(s2 * 2)       # a 20, b 40, c 60
print(s2 + s2)      # a 20, b 40, c 60

# Missing values
s4 = pd.Series([1, np.nan, 3, None, 5])
print(s4.isna())    # 0 False, 1 True, 2 False, 3 True, 4 False
print(s4.dropna())  # Remove NaN values
```

### 7.2.3 DataFrame

A DataFrame is a 2D labeled data structure with columns of potentially different types.

**Creating DataFrames:**

```python
# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Salary': [70000, 85000, 90000, 75000]
}
df = pd.DataFrame(data)
print(df)
#       Name  Age     City  Salary
# 0    Alice   25 New York   70000
# 1      Bob   30   London   85000
# 2  Charlie   35    Paris   90000
# 3    Diana   28    Tokyo   75000

# From NumPy array
arr = np.array([[1, 2], [3, 4], [5, 6]])
df2 = pd.DataFrame(arr, columns=['A', 'B'])
print(df2)
#    A  B
# 0  1  2
# 1  3  4
# 2  5  6

# From CSV (covered later)
# df = pd.read_csv('data.csv')
```

**DataFrame Information:**

```python
print(df.head(2))   # First 2 rows
print(df.tail(2))   # Last 2 rows
print(df.info())    # Summary info
print(df.describe()) # Statistics for numerical columns
print(df.shape)     # (4, 4)
print(df.columns)   # Index(['Name', 'Age', 'City', 'Salary'], dtype='object')
print(df.dtypes)    # Data types of each column
```

### 7.2.4 Selecting and Filtering Data

```python
# Selecting columns
print(df['Name'])        # Single column (Series)
print(df[['Name', 'Age']])  # Multiple columns

# Selecting rows
print(df.iloc[0])        # First row (by position)
print(df.iloc[1:3])      # Rows 1-2
print(df.loc[1:2])       # Rows by label (inclusive)

# Selecting by label
print(df.loc[0, 'Name'])  # Alice
print(df.loc[0:2, ['Name', 'Age']])

# Boolean indexing
print(df[df['Age'] > 30])  # Rows where Age > 30

# Multiple conditions
print(df[(df['Age'] > 25) & (df['Salary'] > 80000)])
#    Name  Age  City  Salary
# 1   Bob   30  London  85000
# 2 Charlie 35  Paris   90000

# Query method
print(df.query('Age > 25 and Salary > 80000'))

# isin
print(df[df['City'].isin(['New York', 'Tokyo'])])

# String methods
print(df[df['Name'].str.startswith('A')])  # Names starting with 'A'
```

### 7.2.5 Adding and Modifying Data

```python
# Adding a column
df['Bonus'] = df['Salary'] * 0.1
print(df)
#       Name  Age     City  Salary   Bonus
# 0    Alice   25 New York   70000  7000.0
# 1      Bob   30   London   85000  8500.0
# 2  Charlie   35    Paris   90000  9000.0
# 3    Diana   28    Tokyo   75000  7500.0

# Adding from calculation
df['Total'] = df['Salary'] + df['Bonus']
print(df)

# Modifying existing column
df['Age'] = df['Age'] + 1
print(df)

# Adding a row
new_row = {'Name': 'Eve', 'Age': 29, 'City': 'Berlin', 'Salary': 80000,
           'Bonus': 8000, 'Total': 88000}
df = df.append(new_row, ignore_index=True)

# Or using loc (if index exists)
# df.loc[4] = ['Eve', 29, 'Berlin', 80000, 8000, 88000]

# Removing columns
df = df.drop('Bonus', axis=1)
print(df)

# Removing rows
df = df.drop(4, axis=0)  # Remove row with index 4
```

### 7.2.6 Data Cleaning

```python
# Create a messy DataFrame
messy = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': ['x', 'y', 'z', 'z'],
    'D': [1, 2, 3, 4]
})
print(messy)
#      A    B  C  D
# 0  1.0  5.0  x  1
# 1  2.0  NaN  y  2
# 2  NaN  NaN  z  3
# 3  4.0  8.0  z  4

# Check for missing values
print(messy.isna())
print(messy.isna().sum())  # Count missing per column

# Drop rows/columns with missing values
print(messy.dropna())      # Drop rows with any NaN
print(messy.dropna(axis=1)) # Drop columns with any NaN
print(messy.dropna(thresh=2))  # Rows with at least 2 non-NA values

# Fill missing values
print(messy.fillna(0))     # Fill with 0
print(messy.fillna(method='ffill'))  # Forward fill
print(messy['B'].fillna(messy['B'].mean()))  # Fill with mean

# Remove duplicates
print(messy.drop_duplicates(subset=['C']))  # Remove duplicate C values

# Rename columns
df = df.rename(columns={'Name': 'FullName'})
print(df)

# Type conversion
df['Age'] = df['Age'].astype('int64')
```

### 7.2.7 Grouping and Aggregation

```python
# Create grouped data
data = {
    'Department': ['Engineering', 'Engineering', 'Marketing', 'Marketing', 'Sales', 'Sales'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Salary': [80000, 75000, 65000, 70000, 60000, 55000],
    'Experience': [5, 3, 4, 6, 2, 4]
}
df = pd.DataFrame(data)

# Group by department
grouped = df.groupby('Department')

# Aggregation
print(grouped['Salary'].mean())
# Department
# Engineering    77500.0
# Marketing      67500.0
# Sales          57500.0
# Name: Salary, dtype: float64

print(grouped['Salary'].agg(['mean', 'min', 'max', 'count']))
#              mean    min    max  count
# Department
# Engineering  77500  75000  80000      2
# Marketing    67500  65000  70000      2
# Sales        57500  55000  60000      2

# Multiple aggregations
print(grouped.agg({
    'Salary': ['mean', 'sum'],
    'Experience': ['mean', 'max']
}))

# Group by multiple columns
grouped = df.groupby(['Department', 'Experience'])
print(grouped['Salary'].mean())

# Transform (returns same shape)
df['Salary_normalized'] = grouped['Salary'].transform(lambda x: (x - x.mean()) / x.std())
print(df)

# Pivot tables
pivot = pd.pivot_table(df, values='Salary', index='Department', columns='Experience', aggfunc='mean')
print(pivot)
```

### 7.2.8 Merging and Joining

```python
# Create two DataFrames
employees = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Department': ['Engineering', 'Marketing', 'Sales', 'Engineering']
})

salaries = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Salary': [80000, 65000, 60000, 75000],
    'Bonus': [5000, 3000, 2000, 4000]
})

# Inner join (default)
merged = pd.merge(employees, salaries, on='ID')
print(merged)
#    ID     Name   Department  Salary  Bonus
# 0   1    Alice  Engineering   80000   5000
# 1   2      Bob    Marketing   65000   3000
# 2   3  Charlie        Sales   60000   2000
# 3   4    Diana  Engineering   75000   4000

# Left join
left_join = pd.merge(employees, salaries, on='ID', how='left')

# Right join
right_join = pd.merge(employees, salaries, on='ID', how='right')

# Outer join
outer_join = pd.merge(employees, salaries, on='ID', how='outer')

# Join on different columns
department_info = pd.DataFrame({
    'Dept': ['Engineering', 'Marketing', 'Sales'],
    'Location': ['Building A', 'Building B', 'Building C']
})
merged_with_dept = pd.merge(employees, department_info, left_on='Department', right_on='Dept')

# Concatenation
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

concat_rows = pd.concat([df1, df2], axis=0)  # Vertically
concat_cols = pd.concat([df1, df2], axis=1)  # Horizontally
```

### 7.2.9 Applying Functions

```python
def double_salary(salary):
    return salary * 2

def categorize_salary(salary):
    if salary > 70000:
        return 'High'
    elif salary > 60000:
        return 'Medium'
    else:
        return 'Low'

# Apply to column
df['Salary_Doubled'] = df['Salary'].apply(double_salary)
print(df)

# Apply to entire DataFrame
df['Category'] = df['Salary'].apply(categorize_salary)
print(df[['Salary', 'Category']])

# Apply to multiple columns
def calculate_bonus(row):
    return row['Salary'] * 0.1 if row['Experience'] > 3 else row['Salary'] * 0.05

df['Bonus'] = df.apply(calculate_bonus, axis=1)
print(df)

# Vectorized operations (much faster)
df['Salary_Plus'] = df['Salary'] * 1.1

# Map (for Series)
dept_mapping = {'Engineering': 'Tech', 'Marketing': 'Business', 'Sales': 'Business'}
df['Dept_Category'] = df['Department'].map(dept_mapping)
print(df)
```

### 7.2.10 Handling Dates and Times

```python
# Creating datetime columns
df = pd.DataFrame({
    'Date': ['2024-01-15', '2024-02-20', '2024-03-25'],
    'Value': [100, 200, 150]
})
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'])

# Extract components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.day_name()
print(df)

# Date range
date_range = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print(date_range)

# Resampling time series
time_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'Value': np.random.randn(30)
})
time_data.set_index('Date', inplace=True)

monthly_avg = time_data.resample('M').mean()  # Monthly averages
quarterly_sum = time_data.resample('Q').sum()  # Quarterly sums
```

---

## 7.3 Matplotlib and Data Visualization

Matplotlib is the foundational plotting library in Python. It provides a MATLAB-like interface and integrates well with NumPy and Pandas.

### 7.3.1 Installation and Import

```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

### 7.3.2 Basic Plotting

```python
# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot
ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2)

# Customize
ax.set_title('Trigonometric Functions', fontsize=16)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

plt.show()
```

### 7.3.3 Common Plot Types

**Scatter Plot:**

```python
# Generate data
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, alpha=0.6, color='purple', s=50)
ax.set_title('Scatter Plot with Correlation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True, alpha=0.3)

# Add a trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, p(x_line), 'r--', label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
ax.legend()
plt.show()
```

**Histogram:**

```python
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_title('Histogram of Normally Distributed Data')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.axvline(x=np.mean(data), color='red', linestyle='--', label=f'Mean = {np.mean(data):.2f}')
ax.axvline(x=np.median(data), color='green', linestyle='--', label=f'Median = {np.median(data):.2f}')
ax.legend()
plt.show()
```

**Bar Chart:**

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 55, 20]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
ax.set_title('Bar Chart Example')
ax.set_xlabel('Category')
ax.set_ylabel('Value')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}',
            ha='center', va='bottom')
plt.show()
```

**Box Plot:**

```python
# Generate grouped data
np.random.seed(42)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(2, 0.5, 100)
data3 = np.random.normal(-1, 0.8, 100)

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([data1, data2, data3], labels=['Group A', 'Group B', 'Group C'])
ax.set_title('Box Plot Comparison')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.3)
plt.show()
```

**Heatmap:**

```python
# Create correlation matrix
np.random.seed(42)
data = np.random.randn(100, 5)
df_heat = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
corr = df_heat.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
plt.colorbar(im, ax=ax)

# Add labels
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Add text annotations
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

ax.set_title('Correlation Heatmap')
plt.show()
```

### 7.3.4 Subplots

```python
# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Generate data
x = np.linspace(0, 10, 100)

# Plot 1: Sine
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Function')
axes[0, 0].grid(True)

# Plot 2: Cosine
axes[0, 1].plot(x, np.cos(x), 'r-', linewidth=2)
axes[0, 1].set_title('Cosine Function')
axes[0, 1].grid(True)

# Plot 3: Exponential
axes[1, 0].plot(x, np.exp(x/5), 'g-', linewidth=2)
axes[1, 0].set_title('Exponential Function')
axes[1, 0].grid(True)

# Plot 4: Scatter
scatter_x = np.random.randn(50)
scatter_y = np.random.randn(50)
axes[1, 1].scatter(scatter_x, scatter_y, alpha=0.6)
axes[1, 1].set_title('Scatter Plot')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

### 7.3.5 Plotting from Pandas

```python
# Create DataFrame
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'Sales': np.random.randn(100).cumsum() + 100,
    'Costs': np.random.randn(100).cumsum() + 80,
    'Profit': np.random.randn(100).cumsum() + 20
})

# Set index
df.set_index('Date', inplace=True)

# Plot all columns
df.plot(figsize=(12, 6))
plt.title('Financial Data Over Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# Specific plot types
df['Sales'].plot(kind='hist', bins=30, alpha=0.7)
plt.show()

# Scatter plot
df.plot.scatter(x='Sales', y='Profit', alpha=0.6)
plt.show()

# Box plot by month
df['Month'] = df.index.month
df.boxplot(column='Sales', by='Month')
plt.show()
```

### 7.3.6 Saving Figures

```python
# Save in different formats
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
ax.set_title('Sine Wave')

fig.savefig('sine.png', dpi=300, bbox_inches='tight')
fig.savefig('sine.pdf', bbox_inches='tight')
fig.savefig('sine.svg', bbox_inches='tight')
```

### 7.3.7 Styling

```python
# Available styles
print(plt.style.available)

# Use a style
plt.style.use('seaborn-v0_8-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax.set_title('Styled Plot', fontsize=16)
ax.legend()
plt.show()

# Custom style
plt.style.use('default')  # Reset to default
```

---

## 7.4 Working with APIs

APIs (Application Programming Interfaces) allow programs to interact with web services and fetch data.

### 7.4.1 Making HTTP Requests with `requests`

```bash
pip install requests
```

```python
import requests
import json

# GET request
response = requests.get('https://api.github.com/repos/pandas-dev/pandas')
if response.status_code == 200:
    data = response.json()
    print(f"Repository: {data['name']}")
    print(f"Stars: {data['stargazers_count']}")
    print(f"Forks: {data['forks_count']}")
else:
    print(f"Error: {response.status_code}")

# GET with parameters
params = {'q': 'python', 'page': 1}
response = requests.get('https://api.github.com/search/repositories', params=params)
if response.status_code == 200:
    data = response.json()
    print(f"Total results: {data['total_count']}")
    for repo in data['items'][:3]:
        print(f"  {repo['name']}")

# POST request (for APIs that require it)
payload = {'key1': 'value1', 'key2': 'value2'}
response = requests.post('https://httpbin.org/post', data=payload)
print(response.json())

# Headers
headers = {
    'User-Agent': 'MyApp/1.0',
    'Accept': 'application/json'
}
response = requests.get('https://api.github.com/repos/pandas-dev/pandas', headers=headers)

# Authentication
# Basic auth
auth = ('username', 'password')
# response = requests.get('https://api.example.com/data', auth=auth)

# Bearer token
headers = {'Authorization': 'Bearer YOUR_TOKEN_HERE'}
# response = requests.get('https://api.example.com/data', headers=headers)
```

### 7.4.2 Working with API Responses

```python
# Parse JSON response
response = requests.get('https://api.github.com/repos/pandas-dev/pandas')
data = response.json()

# Access nested data
print(data['owner']['login'])
print(data['license']['name'])

# Handle pagination
def fetch_all_pages(base_url, params=None):
    results = []
    page = 1
    while True:
        params = params or {}
        params['page'] = page
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        results.extend(data)
        page += 1
    return results

# Example: Fetch all pandas contributors
contributors = fetch_all_pages('https://api.github.com/repos/pandas-dev/pandas/contributors')
print(f"Total contributors: {len(contributors)}")
```

### 7.4.3 Rate Limiting and Error Handling

```python
import time

def api_call_with_retry(url, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', delay))
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: {response.status_code}")
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(delay)
    return None

# Example
result = api_call_with_retry('https://api.github.com/repos/pandas-dev/pandas')
if result:
    print(f"Repository: {result['name']}")
```

### 7.4.4 Working with Common Data Formats

**JSON:**
```python
import json

# Parse JSON string
json_string = '{"name": "Alice", "age": 30}'
data = json.loads(json_string)
print(data['name'])

# Convert to JSON string
data = {"name": "Bob", "age": 25}
json_string = json.dumps(data)
print(json_string)

# Pretty print
print(json.dumps(data, indent=2))

# Write to file
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read from file
with open('data.json', 'r') as f:
    data = json.load(f)
```

**XML:**
```python
import xml.etree.ElementTree as ET

# Parse XML string
xml_string = '<person><name>Alice</name><age>30</age></person>'
root = ET.fromstring(xml_string)
print(root.find('name').text)  # Alice

# Parse XML file
tree = ET.parse('data.xml')
root = tree.getroot()
for child in root:
    print(child.tag, child.attrib)
```

---

## 7.5 Testing and Debugging

### 7.5.1 Testing with `pytest`

```bash
pip install pytest
```

```python
# math_functions.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# test_math_functions.py
import pytest
from math_functions import add, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(6, 2) == 3
    assert divide(5, 2) == 2.5

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(5, 0)

# Run tests: pytest test_math_functions.py -v
```

### 7.5.2 Assertions

```python
# Basic assertions
assert 2 + 2 == 4
assert 3 > 1
assert 'hello' in 'hello world'

# With error message
assert 2 + 2 == 5, "Math doesn't work!"

# Assertions in functions
def process_data(data):
    assert data is not None, "Data cannot be None"
    assert len(data) > 0, "Data cannot be empty"
    return len(data)
```

### 7.5.3 Debugging

**Using print statements:**

```python
def compute_average(numbers):
    print(f"Input: {numbers}")
    total = sum(numbers)
    print(f"Sum: {total}")
    count = len(numbers)
    print(f"Count: {count}")
    if count == 0:
        return 0
    avg = total / count
    print(f"Average: {avg}")
    return avg

compute_average([1, 2, 3, 4, 5])
```

**Using `pdb` (Python Debugger):**

```python
import pdb

def buggy_function(x):
    result = 0
    for i in range(x):
        # pdb.set_trace()  # Breakpoint
        result += i
    return result

# Run with breakpoint
# pdb.set_trace()
# buggy_function(10)

# Then use commands:
# n (next line)
# s (step into)
# c (continue)
# p variable (print variable)
# q (quit)
```

**Using `logging`:**

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)

def process_data(data):
    logging.info(f"Processing data: {data}")
    try:
        result = data['value'] * 2
        logging.debug(f"Result: {result}")
        return result
    except KeyError as e:
        logging.error(f"Key error: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return None

process_data({'value': 10})
process_data({})
```

### 7.5.4 Profiling Code

```python
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
slow_function()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10

# Using timeit
import timeit

def fast_function():
    return sum(range(1000000))

print(timeit.timeit(slow_function, number=10))
print(timeit.timeit(fast_function, number=10))
```

---

## 7.6 End-to-End Example: Data Analysis Pipeline

Let's combine everything into a complete data analysis pipeline:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAnalyzer:
    """Complete data analysis pipeline class."""

    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.stats = {}

    def fetch_data_from_api(self, url: str, params: Dict = None) -> Dict[str, Any]:
        """Fetch data from an API."""
        try:
            logging.info(f"Fetching data from {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Successfully fetched data")
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logging.info(f"Loading data from {filename}")
            self.data = pd.read_csv(filename)
            logging.info(f"Loaded {len(self.data)} records")
            return self.data
        except FileNotFoundError:
            logging.error(f"File {filename} not found")
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        logging.info("Cleaning data...")
        cleaned = df.copy()

        # Remove duplicates
        cleaned = cleaned.drop_duplicates()
        logging.info(f"Removed duplicates, {len(cleaned)} records remaining")

        # Handle missing values
        for col in cleaned.columns:
            if cleaned[col].isna().sum() > 0:
                if cleaned[col].dtype in ['int64', 'float64']:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    logging.info(f"Filled missing values in {col} with median")
                else:
                    cleaned[col] = cleaned[col].fillna('Unknown')
                    logging.info(f"Filled missing values in {col} with 'Unknown'")

        # Remove outliers (using IQR method)
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned = cleaned[(cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)]

        self.cleaned_data = cleaned
        logging.info(f"Cleaning complete, {len(cleaned)} records remaining")
        return cleaned

    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """Perform statistical analysis."""
        logging.info("Analyzing data...")
        stats = {}

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats[col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }

        self.stats = stats
        logging.info("Analysis complete")
        return stats

    def create_visualizations(self, df: pd.DataFrame, save_dir: str = 'plots'):
        """Create and save visualizations."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Histogram
            axes[0].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[0].set_title(f'Distribution of {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            axes[0].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[0].axvline(df[col].median(), color='green', linestyle='--', label='Median')
            axes[0].legend()

            # Box plot
            axes[1].boxplot(df[col])
            axes[1].set_title(f'Box Plot of {col}')
            axes[1].set_ylabel(col)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/{col}_distribution.png', dpi=150)
            plt.close()

        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=150)
            plt.close()

        logging.info(f"Visualizations saved to {save_dir}/")

    def generate_report(self, df: pd.DataFrame, stats: Dict) -> str:
        """Generate a text report of the analysis."""
        report = []
        report.append("=" * 60)
        report.append("DATA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nDataset Summary:")
        report.append(f"  Total records: {len(df)}")
        report.append(f"  Total features: {len(df.columns)}")
        report.append(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        report.append("\nNumeric Features:")
        for col, stat in stats.items():
            if 'mean' in stat:  # Numeric feature
                report.append(f"\n  {col}:")
                report.append(f"    Mean: {stat['mean']:.2f}")
                report.append(f"    Median: {stat['median']:.2f}")
                report.append(f"    Std Dev: {stat['std']:.2f}")
                report.append(f"    Range: {stat['min']:.2f} - {stat['max']:.2f}")
                report.append(f"    Skew: {stat['skew']:.2f}")

        report.append("\nCategorical Features:")
        for col, stat in stats.items():
            if 'unique_count' in stat:  # Categorical feature
                report.append(f"\n  {col}:")
                report.append(f"    Unique values: {stat['unique_count']}")
                report.append(f"    Top values: {stat['top_values']}")

        report.append("\n" + "=" * 60)

        # Save report
        with open('analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))

        logging.info("Report generated and saved to analysis_report.txt")
        return '\n'.join(report)

    def run_pipeline(self, filename: str):
        """Run the complete data analysis pipeline."""
        logging.info("=" * 60)
        logging.info("Starting data analysis pipeline")
        logging.info("=" * 60)

        # 1. Load data
        df = self.load_from_csv(filename)
        if df is None:
            return

        # 2. Clean data
        df_cleaned = self.clean_data(df)

        # 3. Analyze data
        stats = self.analyze_data(df_cleaned)

        # 4. Create visualizations
        self.create_visualizations(df_cleaned)

        # 5. Generate report
        report = self.generate_report(df_cleaned, stats)

        logging.info("=" * 60)
        logging.info("Pipeline complete!")
        return df_cleaned, stats, report

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.randint(18, 65, 200),
        'Salary': np.random.normal(60000, 15000, 200),
        'Experience': np.random.randint(0, 40, 200),
        'Department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], 200),
        'Performance': np.random.uniform(1, 5, 200)
    })

    # Add some missing values
    sample_data.loc[np.random.choice(200, 10), 'Age'] = np.nan
    sample_data.loc[np.random.choice(200, 5), 'Salary'] = np.nan

    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)

    # Run pipeline
    analyzer = DataAnalyzer()
    df, stats, report = analyzer.run_pipeline('sample_data.csv')

    print("\n" + report)
```

---

## Summary

This chapter covered the essential scientific Python stack for AI development:

- **NumPy:** The foundation for numerical computing with n-dimensional arrays, vectorized operations, broadcasting, and linear algebra.
- **Pandas:** High-level data manipulation and analysis with DataFrames, including data cleaning, grouping, merging, and time series.
- **Matplotlib:** Visualization library for creating publication-quality plots, including line plots, scatter plots, histograms, and heatmaps.
- **APIs:** Fetching and working with web data using the `requests` library.
- **Testing and Debugging:** Writing tests with pytest, debugging with pdb and logging, and profiling code.

These tools form the core of the data science workflow and will be used extensively throughout the rest of this book.

---

##  Further Reading & Resources

- **NumPy Documentation:** [numpy.org/doc/](https://numpy.org/doc/)
- **Pandas Documentation:** [pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **Matplotlib Gallery:** [matplotlib.org/gallery/](https://matplotlib.org/gallery/)
- **Books:**
  - *Python for Data Analysis* by Wes McKinney (Pandas creator).
  - *Data Science Handbook* by Jake VanderPlas.

---

##  Chapter 7 Checklist

Before moving on, ensure you can:

- [ ] Create and manipulate NumPy arrays (creation, indexing, slicing, operations).
- [ ] Use vectorized operations for performance.
- [ ] Apply broadcasting rules.
- [ ] Create and manipulate Pandas Series and DataFrames.
- [ ] Perform data cleaning (missing values, duplicates, outliers).
- [ ] Use grouping and aggregation with Pandas.
- [ ] Merge and join DataFrames.
- [ ] Create various plots with Matplotlib (line, scatter, histogram, bar, heatmap).
- [ ] Customize plots (titles, labels, legends, colors).
- [ ] Save and export visualizations.
- [ ] Fetch data from APIs and handle JSON responses.
- [ ] Write unit tests for code.
- [ ] Debug effectively using pdb and logging.

---

##  Hands-On Exercises

1. **NumPy Operations:**
   - Create a 5x5 matrix with random integers between 1 and 100.
   - Compute the mean, median, and standard deviation.
   - Normalize the matrix (subtract mean, divide by std).
   - Find the indices of the top 5 largest values.

2. **Pandas Data Manipulation:**
   - Load the Iris dataset (available in Seaborn) into a DataFrame.
   - Compute summary statistics for each species.
   - Create a new column for petal area (petal_length * petal_width).
   - Group by species and compute mean values for all numeric columns.
   - Filter for rows where sepal_length > 5.0.

3. **Data Cleaning Challenge:**
   - Create a DataFrame with 100 rows and 5 columns containing random data, with intentional missing values (10% of values).
   - Write a function that:
     - Fills missing numeric values with the column median.
     - Fills missing categorical values with the mode.
     - Removes rows with more than 3 missing values.
     - Detects and removes outliers (IQR method).

4. **Visualization Exercise:**
   - Load the Titanic dataset (from Seaborn).
   - Create the following visualizations:
     - Survival rate by passenger class (bar chart).
     - Age distribution of passengers (histogram).
     - Fare by passenger class (box plot).
     - Correlation heatmap of numeric features.
     - Pair plot of selected features.

5. **API Integration:**
   - Write a script that:
     - Fetches weather data from a public API (e.g., OpenWeatherMap).
     - Parses and structures the data in a DataFrame.
     - Creates a time series plot of temperature over time.
     - Saves the data to CSV.

6. **Testing Practice:**
   - Write a class with methods for common statistical calculations (mean, median, variance, correlation).
   - Write unit tests for each method using pytest.
   - Test edge cases (empty lists, single values, invalid inputs).

7. **Complete Pipeline Project:**
   - Choose a dataset (e.g., from Kaggle or public data sources).
   - Build an end-to-end data analysis pipeline that:
     - Loads and cleans the data.
     - Performs exploratory data analysis.
     - Creates visualizations.
     - Generates a summary report.
     - Exports cleaned data to CSV and JSON formats.
