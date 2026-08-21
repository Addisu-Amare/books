# Chapter 6: Python Programming Fundamentals

## Chapter Overview

Python has become the dominant programming language for artificial intelligence and machine learning. Its readability, extensive ecosystem of libraries, and vibrant community make it the ideal choice for both beginners and experts. This chapter provides a comprehensive introduction to Python programming, covering everything from basic syntax to object-oriented programming. By the end of this chapter, you'll be able to write clean, efficient Python code and understand the language features that make it so powerful for AI development.

**Learning Objectives:**
By the end of this chapter, you will be able to:
- Set up a Python development environment and write basic Python scripts.
- Understand and use Python's fundamental data types and operations.
- Implement control flow using conditions and loops.
- Define and use functions to write reusable code.
- Work with Python's core data structures: lists, tuples, sets, and dictionaries.
- Read from and write to files.
- Handle errors and exceptions gracefully.
- Apply object-oriented programming concepts (classes, objects, inheritance).
- Organize code using modules and packages.

---

## 6.1 Setting Up Your Python Environment

### 6.1.1 Installing Python

Python can be downloaded from the official website: [python.org](https://python.org).

**Windows:**
1. Download the installer from python.org.
2. Run the installer.
3. **Important:** Check "Add Python to PATH" during installation.
4. Verify installation: Open Command Prompt and type `python --version`.

**macOS:**
```bash
# Install using Homebrew (recommended)
brew install python3

# Or download from python.org
# Verify installation
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

### 6.1.2 The Python Interactive Shell

Python can be used interactively, which is great for experimentation:

```bash
python3
```

```python
>>> print("Hello, World!")
Hello, World!
>>> 2 + 3
5
>>> exit()
```

### 6.1.3 Running Python Scripts

Create a file with a `.py` extension and run it:

```bash
# my_script.py
print("Hello from a script!")

# Run from terminal
python3 my_script.py
```

### 6.1.4 Virtual Environments

Virtual environments isolate project dependencies—essential for AI projects with many packages.

**Creating a virtual environment:**
```bash
# Create environment
python3 -m venv myenv

# Activate environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate

# Install packages
pip install numpy pandas matplotlib

# Deactivate
deactivate
```

### 6.1.5 Choosing a Development Environment

- **VS Code:** Free, lightweight, excellent Python support with extensions.
- **PyCharm:** Full-featured IDE with great Python-specific tools (Community edition is free).
- **Jupyter Notebook:** Interactive environment ideal for data exploration and visualization.
- **IDLE:** Simple editor that comes with Python.

For this book, we recommend **VS Code** with the Python extension.

---

## 6.2 Variables, Data Types, and Basic Operations

### 6.2.1 Variables

Variables store values. Python is dynamically typed—you don't need to declare the type.

```python
# Variable assignment
name = "Alice"          # string
age = 30                # integer
height = 5.8            # float
is_student = True       # boolean

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Python is case-sensitive
myVar = 10
myvar = 20  # Different variable
```

### 6.2.2 Naming Conventions

- Use lowercase with underscores: `my_variable`
- Descriptive names: `student_name` rather than `sn`
- Constants use uppercase: `MAX_ITERATIONS = 1000`
- Avoid Python keywords: `if`, `for`, `while`, etc.

### 6.2.3 Basic Data Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | Integer | `42` |
| `float` | Floating-point number | `3.14159` |
| `bool` | Boolean (True/False) | `True`, `False` |
| `str` | String | `"Hello"`, `'World'` |
| `NoneType` | None value | `None` |

**Type Checking and Conversion:**
```python
# Check type
x = 42
print(type(x))  # <class 'int'>

# Type conversion
x = "123"
y = int(x)      # 123 (int)
z = float(x)    # 123.0 (float)
s = str(42)     # "42" (string)
b = bool(1)     # True

# Type conversion in operations
result = 10 + 3.5  # 13.5 (float)
```

### 6.2.4 Basic Operations

**Arithmetic Operations:**
```python
a = 10
b = 3

print(a + b)   # 13  (addition)
print(a - b)   # 7   (subtraction)
print(a * b)   # 30  (multiplication)
print(a / b)   # 3.333... (float division)
print(a // b)  # 3   (integer division, floors the result)
print(a % b)   # 1   (modulo, remainder)
print(a ** b)  # 1000 (exponentiation)
```

**String Operations:**
```python
name = "Alice"
greeting = "Hello"

# Concatenation
message = greeting + ", " + name + "!"  # "Hello, Alice!"

# Repetition
stars = "*" * 10  # "**********"

# Length
print(len(name))  # 5

# Indexing (0-based)
print(name[0])    # 'A'
print(name[-1])   # 'e' (last character)

# Slicing [start:end:step]
text = "Python Programming"
print(text[0:6])   # "Python"
print(text[7:])    # "Programming"
print(text[::-1])  # "gnimmargorP nohtyP" (reversed)

# String methods
print(text.lower())  # "python programming"
print(text.upper())  # "PYTHON PROGRAMMING"
print(text.split())  # ['Python', 'Programming']
print(text.replace("Python", "AI"))  # "AI Programming"
```

**F-Strings (Formatted Strings):**
Python 3.6+ supports f-strings for easy string formatting:

```python
name = "Bob"
age = 25
height = 1.75

print(f"Name: {name}, Age: {age}, Height: {height:.2f}m")
# "Name: Bob, Age: 25, Height: 1.75m"

# Expressions in f-strings
print(f"Next year, {name} will be {age + 1}")
```

---

## 6.3 Control Flow: Conditions and Loops

### 6.3.1 Conditional Statements

**The `if` Statement:**

```python
age = 18

if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")
```

**`if-elif-else`:**

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")  # "Grade: B"
```

**Comparison Operators:**
- `==` equal to
- `!=` not equal to
- `>` greater than
- `<` less than
- `>=` greater than or equal to
- `<=` less than or equal to

**Logical Operators:**
- `and` (True if both true)
- `or` (True if at least one true)
- `not` (negates condition)

```python
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive.")

if not has_license:
    print("You need a license.")

temperature = 30
is_sunny = True

if temperature > 25 or is_sunny:
    print("It's warm or sunny!")
```

**Membership Operators:**
```python
fruits = ["apple", "banana", "orange"]

print("apple" in fruits)    # True
print("grape" in fruits)    # False
print("banana" not in fruits)  # False
```

### 6.3.2 Loops

**The `for` Loop:**

```python
# Iterate over a list
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# Iterate over a string
for char in "Python":
    print(char)

# Using range()
for i in range(5):       # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8):    # 2, 3, 4, 5, 6, 7
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (step=2)
    print(i)
```

**The `while` Loop:**

```python
# Countdown
count = 5
while count > 0:
    print(count)
    count -= 1
print("Blast off!")

# Infinite loop (use carefully!)
# while True:
#     print("Press Ctrl+C to stop")
```

**Loop Control:**
- `break`: Exit the loop entirely.
- `continue`: Skip the rest of the current iteration.
- `else`: Execute after loop completes normally (without break).

```python
# Break
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# Continue
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4

# Else with loops
for i in range(3):
    print(i)
else:
    print("Loop completed normally")
```

---

## 6.4 Functions

Functions are reusable blocks of code that take inputs (parameters) and return outputs.

### 6.4.1 Defining and Calling Functions

```python
def greet(name):
    """Print a greeting."""
    print(f"Hello, {name}!")

def add(a, b):
    """Return the sum of two numbers."""
    return a + b

def square(x):
    """Return the square of x."""
    return x ** 2

# Calling functions
greet("Alice")          # "Hello, Alice!"
result = add(5, 3)      # 8
squared = square(4)     # 16
```

### 6.4.2 Function Parameters

**Default Parameters:**
```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")          # "Hello, Alice!"
greet("Bob", "Hi")      # "Hi, Bob!"
```

**Keyword Arguments:**
```python
def describe_person(name, age, city="Unknown"):
    print(f"{name} is {age} years old and lives in {city}.")

describe_person("Alice", 30, city="New York")
describe_person(age=25, name="Bob")  # Order doesn't matter with keywords
```

**Variable Number of Arguments:**

```python
def sum_all(*args):
    """Sum any number of positional arguments."""
    return sum(args)

def print_info(**kwargs):
    """Print keyword arguments."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print(sum_all(1, 2, 3, 4, 5))  # 15
print_info(name="Alice", age=30, city="New York")
```

### 6.4.3 Docstrings

Docstrings document functions, classes, and modules.

```python
def multiply(a, b):
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b

    Raises:
        TypeError: If arguments are not numbers
    """
    return a * b

print(multiply.__doc__)  # View the docstring
help(multiply)           # Full documentation
```

### 6.4.4 Lambda Functions

Lambda functions are anonymous, one-line functions.

```python
# Regular function
def square(x):
    return x ** 2

# Lambda equivalent
square_lambda = lambda x: x ** 2

print(square_lambda(5))  # 25

# Used with sorting, filtering, etc.
numbers = [1, 5, 2, 8, 3]
sorted_numbers = sorted(numbers, key=lambda x: -x)  # Sort descending
print(sorted_numbers)  # [8, 5, 3, 2, 1]

# Filtering
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 8]
```

### 6.4.5 Scope and Namespace

```python
x = 10  # Global variable

def my_function():
    y = 5  # Local variable
    print(x)  # Access global variable
    print(y)

my_function()
print(x)  # Works
# print(y)  # Error: y not defined outside function

# Modifying global variables
counter = 0

def increment():
    global counter
    counter += 1

increment()
print(counter)  # 1
```

---

## 6.5 Core Data Structures

### 6.5.1 Lists

Lists are ordered, mutable sequences.

```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Accessing elements (0-based indexing)
print(fruits[0])      # "apple"
print(fruits[-1])     # "orange" (last element)

# Slicing
print(fruits[1:3])    # ["banana", "orange"]
print(fruits[:2])     # ["apple", "banana"]
print(fruits[1:])     # ["banana", "orange"]

# Common methods
fruits.append("grape")      # Add to end
fruits.insert(1, "kiwi")    # Insert at index
fruits.remove("banana")     # Remove first occurrence
popped = fruits.pop()       # Remove and return last
fruits.sort()               # Sort in place
fruits.reverse()            # Reverse in place

# List comprehension (powerful!)
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(20) if x % 2 == 0]  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Nested lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[1][2])  # 6
```

### 6.5.2 Tuples

Tuples are ordered, immutable sequences.

```python
# Creating tuples
point = (3, 4)
colors = ("red", "green", "blue")
single = (5,)  # Note the comma!

# Accessing (same as lists)
print(point[0])   # 3

# Unpacking
x, y = point
print(x, y)       # 3 4

# Tuples are immutable
# point[0] = 5  # Error!

# Useful for returning multiple values from functions
def get_user_info():
    return "Alice", 30, "alice@email.com"

name, age, email = get_user_info()
```

### 6.5.3 Sets

Sets are unordered collections of unique elements.

```python
# Creating sets
colors = {"red", "green", "blue"}
numbers = set([1, 2, 3, 3, 4, 4, 5])  # {1, 2, 3, 4, 5}

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(set1 | set2)   # Union: {1, 2, 3, 4, 5, 6}
print(set1 & set2)   # Intersection: {3, 4}
print(set1 - set2)   # Difference: {1, 2}
print(set1 ^ set2)   # Symmetric difference: {1, 2, 5, 6}

# Methods
set1.add(7)
set1.remove(2)
# set1.remove(10)   # Error if not found
set1.discard(10)     # No error if not found

# Useful for removing duplicates
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(numbers))  # [1, 2, 3, 4]
```

### 6.5.4 Dictionaries

Dictionaries store key-value pairs.

```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing
print(person["name"])    # "Alice"
print(person.get("age")) # 30
print(person.get("country", "Unknown"))  # "Unknown" (default)

# Modifying
person["age"] = 31
person["email"] = "alice@email.com"
del person["city"]

# Keys must be immutable (strings, numbers, tuples)
# Values can be anything

# Iterating
for key in person:
    print(f"{key}: {person[key]}")

for key, value in person.items():
    print(f"{key}: {value}")

for key in person.keys():
    print(key)

for value in person.values():
    print(value)

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Nested dictionaries
employees = {
    "alice": {"age": 30, "department": "Engineering"},
    "bob": {"age": 25, "department": "Marketing"}
}
print(employees["alice"]["age"])  # 30
```

### 6.5.5 Choosing the Right Data Structure

| Structure | Ordered | Mutable | Index | Duplicates | Use Case |
|-----------|---------|---------|-------|------------|----------|
| List | Yes | Yes | Integer | Yes | Ordered collection, frequent modifications |
| Tuple | Yes | No | Integer | Yes | Fixed collection, efficient, as keys |
| Set | No | Yes | No | No | Unique items, membership tests |
| Dict | Yes (3.7+) | Yes | Key | No (keys) | Key-value lookups, structured data |

---

## 6.6 File Handling

### 6.6.1 Reading Files

```python
# Read entire file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Read line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip removes newline

# Read all lines into a list
with open("data.txt", "r") as file:
    lines = file.readlines()

# Read specific lines
with open("data.txt", "r") as file:
    first_line = file.readline()
    second_line = file.readline()
```

### 6.6.2 Writing Files

```python
# Write to file (overwrites)
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.\n")

# Append to file
with open("output.txt", "a") as file:
    file.write("Appended line.\n")

# Writing multiple lines
lines = ["Line 1", "Line 2", "Line 3"]
with open("output.txt", "w") as file:
    file.writelines([line + "\n" for line in lines])
```

### 6.6.3 Reading CSV Files (Common in AI)

```python
import csv

# Reading CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    headers = next(reader)  # First row
    for row in reader:
        print(row)

# Reading CSV as dictionaries
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row["name"], row["age"])

# Writing CSV
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 30, "New York"])
    writer.writerow(["Bob", 25, "London"])
```

### 6.6.4 Working with JSON (Common for APIs)

```python
import json

# Writing JSON
data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "hobbies": ["reading", "swimming"]
}

with open("data.json", "w") as file:
    json.dump(data, file, indent=2)  # Pretty print

# Reading JSON
with open("data.json", "r") as file:
    loaded_data = json.load(file)

# Working with JSON strings
json_string = json.dumps(data, indent=2)
parsed_data = json.loads(json_string)
```

### 6.6.5 Context Managers

The `with` statement ensures resources are properly managed (e.g., files are closed).

```python
# Without context manager (manual cleanup)
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()

# With context manager (automatic cleanup)
with open("data.txt", "r") as file:
    content = file.read()
```

---

## 6.7 Error Handling

### 6.7.1 Try-Except Blocks

```python
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("You can't divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
```

### 6.7.2 Else and Finally

```python
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
else:
    # Runs if no exception occurred
    print("File read successfully!")
finally:
    # Always runs
    print("Cleanup...")
    try:
        file.close()
    except:
        pass  # File may not exist
```

### 6.7.3 Raising Exceptions

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age

try:
    age = validate_age(-5)
except ValueError as e:
    print(f"Validation error: {e}")
```

### 6.7.4 Custom Exceptions

```python
class InvalidInputError(Exception):
    """Custom exception for invalid input."""
    pass

def process_input(value):
    if not isinstance(value, int):
        raise InvalidInputError("Input must be an integer")
    return value * 2

try:
    result = process_input("10")
except InvalidInputError as e:
    print(f"Error: {e}")
```

---

## 6.8 Object-Oriented Programming

OOP is a paradigm that organizes code into classes and objects. It's especially useful for building complex AI systems.

### 6.8.1 Classes and Objects

```python
class Person:
    """A simple Person class."""

    # Class variable (shared by all instances)
    species = "Homo sapiens"

    # Constructor (initializer)
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
        self._secret = "I like cheese"  # Protected attribute

    # Instance method
    def greet(self):
        return f"Hello, my name is {self.name}"

    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! Now I'm {self.age}"

    # String representation
    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Creating objects
alice = Person("Alice", 30)
bob = Person("Bob", 25)

print(alice.greet())        # "Hello, my name is Alice"
print(alice.have_birthday())  # "Happy birthday! Now I'm 31"
print(alice)                # "Person(name=Alice, age=31)"
print(Person.species)       # "Homo sapiens"
```

### 6.8.2 Inheritance

```python
class Student(Person):
    """Student class inheriting from Person."""

    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # Call parent constructor
        self.student_id = student_id
        self.courses = []

    def enroll(self, course):
        self.courses.append(course)
        return f"{self.name} enrolled in {course}"

    def greet(self):
        """Override parent method."""
        return f"Hi, I'm {self.name}, student #{self.student_id}"

    def __str__(self):
        return f"Student(name={self.name}, age={self.age}, id={self.student_id})"

# Using inheritance
student = Student("Charlie", 20, "S12345")
print(student.greet())          # "Hi, I'm Charlie, student #S12345"
print(student.enroll("AI 101"))  # "Charlie enrolled in AI 101"
```

### 6.8.3 Composition

```python
class Address:
    def __init__(self, street, city, zip_code):
        self.street = street
        self.city = city
        self.zip_code = zip_code

    def __str__(self):
        return f"{self.street}, {self.city}, {self.zip_code}"

class Employee:
    def __init__(self, name, position, address):
        self.name = name
        self.position = position
        self.address = address  # Composition (has-a relationship)

    def __str__(self):
        return f"{self.name} ({self.position}), {self.address}"

address = Address("123 Main St", "New York", "10001")
emp = Employee("Diana", "Engineer", address)
print(emp)  # "Diana (Engineer), 123 Main St, New York, 10001"
```

### 6.8.4 Special Methods (Magic Methods)

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Overload + operator."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Overload - operator."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Overload * operator."""
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other):
        """Overload == operator."""
        return self.x == other.x and self.y == other.y

    def __len__(self):
        """Overload len() function."""
        return 2

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
print(v3)  # "Vector(4, 6)"
print(v1 * 3)  # "Vector(3, 6)"
print(v1 == Vector(1, 2))  # True
print(len(v1))  # 2
```

### 6.8.5 Property Decorators

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute

    @property
    def balance(self):
        """Getter for balance."""
        return self.__balance

    @balance.setter
    def balance(self, amount):
        """Setter for balance with validation."""
        if amount < 0:
            raise ValueError("Balance cannot be negative")
        self.__balance = amount

    @property
    def is_rich(self):
        """Read-only property."""
        return self.__balance > 1000000

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.__balance += amount

    def withdraw(self, amount):
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        self.__balance -= amount

account = BankAccount("Alice", 1000)
print(account.balance)  # 1000
account.balance = 2000  # Using setter
account.deposit(500)
print(account.is_rich)  # False
```

---

## 6.9 Modules and Packages

### 6.9.1 Modules

A module is a Python file containing definitions and statements.

**Creating a module** (`math_utils.py`):
```python
# math_utils.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

PI = 3.14159
```

**Importing modules:**
```python
# Import entire module
import math_utils
print(math_utils.add(5, 3))  # 8
print(math_utils.PI)         # 3.14159

# Import specific items
from math_utils import add, PI
print(add(5, 3))  # 8
print(PI)         # 3.14159

# Import with alias
import math_utils as mu
print(mu.subtract(10, 4))  # 6

# Import all (not recommended)
from math_utils import *
print(add(5, 3))
```

### 6.9.2 Standard Library Modules

```python
import math
import random
import datetime
import os
import sys
import re
import collections

# Math
print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.14159

# Random
print(random.randint(1, 10))  # Random integer
print(random.choice(["a", "b", "c"]))  # Random choice

# Datetime
now = datetime.datetime.now()
print(now.year, now.month, now.day)

# OS operations
print(os.getcwd())  # Current working directory
print(os.listdir())  # Files in directory

# Regular expressions
pattern = re.compile(r'\d+')
print(pattern.findall("abc123def456"))  # ['123', '456']
```

### 6.9.3 Creating Packages

A package is a directory containing modules and an `__init__.py` file.

```
my_package/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        module3.py
```

**`__init__.py` can define what's exported:**
```python
# my_package/__init__.py
from .module1 import function1
from .module2 import function2

__all__ = ['function1', 'function2']
```

### 6.9.4 Third-Party Packages (pip)

```bash
# Installing packages
pip install numpy pandas matplotlib

# Install specific version
pip install numpy==1.21.0

# Upgrade packages
pip install --upgrade numpy

# Uninstall
pip uninstall numpy

# List installed packages
pip list

# Freeze dependencies
pip freeze > requirements.txt

# Install from requirements file
pip install -r requirements.txt
```

---

## 6.10 Python for AI: A Practical Example

Let's create a simple data analysis pipeline using the concepts we've learned:

```python
import csv
import json
from dataclasses import dataclass
from typing import List, Optional
import statistics

@dataclass
class Student:
    """Data class for student records."""
    name: str
    age: int
    grades: List[float]
    email: Optional[str] = None

    @property
    def average(self) -> float:
        """Calculate average grade."""
        return statistics.mean(self.grades) if self.grades else 0.0

    @property
    def passed(self) -> bool:
        """Determine if student passed (average >= 60)."""
        return self.average >= 60

class StudentProcessor:
    """Process student data."""

    def __init__(self, students: List[Student]):
        self.students = students

    def get_top_students(self, n: int = 3) -> List[Student]:
        """Get top n students by average grade."""
        return sorted(self.students, key=lambda s: s.average, reverse=True)[:n]

    def get_failing_students(self) -> List[Student]:
        """Get all failing students."""
        return [s for s in self.students if not s.passed]

    def get_statistics(self) -> dict:
        """Get summary statistics."""
        averages = [s.average for s in self.students]
        return {
            "count": len(self.students),
            "mean": statistics.mean(averages),
            "median": statistics.median(averages),
            "std_dev": statistics.stdev(averages) if len(averages) > 1 else 0,
            "passing_count": sum(1 for s in self.students if s.passed)
        }

    def export_to_csv(self, filename: str):
        """Export students to CSV."""
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Age", "Average", "Passed", "Grades"])
            for s in self.students:
                writer.writerow([
                    s.name,
                    s.age,
                    f"{s.average:.2f}",
                    "Yes" if s.passed else "No",
                    ", ".join(str(g) for g in s.grades)
                ])

    def export_to_json(self, filename: str):
        """Export students to JSON."""
        data = [
            {
                "name": s.name,
                "age": s.age,
                "grades": s.grades,
                "average": s.average,
                "passed": s.passed,
                "email": s.email
            }
            for s in self.students
        ]
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)

def load_from_csv(filename: str) -> List[Student]:
    """Load students from CSV."""
    students = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            grades = [float(g.strip()) for g in row['Grades'].split(',')]
            student = Student(
                name=row['Name'],
                age=int(row['Age']),
                grades=grades
            )
            students.append(student)
    return students

# Example usage
def main():
    # Create sample data
    students = [
        Student("Alice", 20, [85, 90, 88, 92], "alice@email.com"),
        Student("Bob", 19, [75, 80, 72, 68]),
        Student("Charlie", 21, [50, 45, 55, 60]),
        Student("Diana", 20, [95, 93, 97, 94]),
        Student("Eve", 19, [30, 40, 35, 45])
    ]

    # Process data
    processor = StudentProcessor(students)

    print("=== Student Statistics ===")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Top 3 Students ===")
    for i, s in enumerate(processor.get_top_students(3), 1):
        print(f"{i}. {s.name}: {s.average:.2f}")

    print("\n=== Failing Students ===")
    for s in processor.get_failing_students():
        print(f"{s.name}: {s.average:.2f}")

    # Export data
    processor.export_to_csv("students.csv")
    processor.export_to_json("students.json")
    print("\nData exported to students.csv and students.json")

    # Load from CSV
    loaded = load_from_csv("students.csv")
    print(f"\nLoaded {len(loaded)} students from CSV")

if __name__ == "__main__":
    main()
```

---

## 6.11 Best Practices and Style Guide

### 6.11.1 PEP 8 Style Guide

Python follows PEP 8 for code style:

```python
# ✅ Good
def calculate_average(grades: List[float]) -> float:
    """Calculate average of grades."""
    return sum(grades) / len(grades) if grades else 0.0

class StudentRecord:
    """Store student information."""

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# ❌ Bad
def CalcAvg(grades):
    return sum(grades)/len(grades) if grades else 0

class studentRecord:
    def __init__(self,name,age):
        self.name=name
        self.age=age
```

**Key PEP 8 Rules:**
- 4 spaces per indentation level.
- Maximum line length: 79 characters (99 for comments/docstrings).
- Imports at the top of the file, grouped (standard library, third-party, local).
- Class names in `CapWords` (e.g., `StudentRecord`).
- Function/variable names in `snake_case` (e.g., `calculate_average`).
- Constants in `UPPER_CASE` (e.g., `MAX_ITERATIONS`).

### 6.11.2 Type Hints

Python 3.5+ supports type hints:

```python
from typing import List, Optional, Union, Dict, Tuple

def process_data(data: List[int]) -> Tuple[float, int]:
    """Process list of integers."""
    return sum(data) / len(data), len(data)

def get_user(user_id: int) -> Optional[Dict[str, Union[str, int]]]:
    """Get user by ID, returns None if not found."""
    # ...
    return {"name": "Alice", "age": 30}

# Type checking (requires mypy)
# pip install mypy
# mypy script.py
```

### 6.11.3 Docstrings

Use docstrings to document functions, classes, and modules:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Extended description with more details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this error occurs.

    Examples:
        >>> function_name("hello", 5)
        True
    """
    pass
```

---

## Summary

This chapter covered Python programming fundamentals essential for AI development:

- **Development environment** setup, including virtual environments.
- **Variables and data types** (int, float, bool, str, None).
- **Control flow** with if-elif-else and loops (for, while).
- **Functions** for reusable code, including lambda functions.
- **Core data structures:** lists, tuples, sets, and dictionaries.
- **File handling** for reading/writing CSV, JSON, and text files.
- **Error handling** with try-except-else-finally.
- **Object-oriented programming** with classes, inheritance, and composition.
- **Modules and packages** for organizing code.

These fundamentals will be used throughout the rest of the book as we build more advanced AI systems.

---

##  Further Reading & Resources

- **Books:**
  - *Python Crash Course* by Eric Matthes.
  - *Automate the Boring Stuff with Python* by Al Sweigart.
  - *Effective Python* by Brett Slatkin.
- **Online:**
  - [Python.org Tutorial](https://docs.python.org/3/tutorial/)
  - [Real Python](https://realpython.com/)
  - [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
  - [Python Cheat Sheet](https://www.pythoncheatsheet.org/)

---

##  Chapter 6 Checklist

Before moving on, ensure you can:

- [ ] Set up a Python virtual environment and install packages.
- [ ] Create and run Python scripts.
- [ ] Work with all basic data types and operations.
- [ ] Use if-elif-else statements and loops effectively.
- [ ] Define functions with parameters and return values.
- [ ] Work with lists, tuples, sets, and dictionaries.
- [ ] Read from and write to files.
- [ ] Handle exceptions and errors gracefully.
- [ ] Define and use classes and objects.
- [ ] Import and use modules and packages.

---

## Hands-On Exercises

1. **String Manipulation:**
   - Write a function that counts the number of vowels and consonants in a string.
   - Write a function that checks if a string is a palindrome.

2. **List Operations:**
   - Write a function that returns the second largest element in a list.
   - Write a function that merges two sorted lists into a single sorted list.

3. **Dictionary Practice:**
   - Write a program that counts word frequencies in a text.
   - Create a simple phonebook using dictionaries (add, search, delete contacts).

4. **File I/O:**
   - Read a CSV file containing student data, compute averages, and write results to a new CSV.
   - Parse a JSON file and extract specific fields.

5. **OOP:**
   - Create a `BankAccount` class with methods for deposit, withdraw, and displaying balance.
   - Create a `Course` class and a `Student` class with enrollment/grade management.

6. **Data Processing:**
   - Build a simple data analysis tool that:
     - Reads data from CSV.
     - Computes summary statistics.
     - Filters data based on conditions.
     - Exports results to JSON.

7. **Error Handling:**
   - Write a robust CSV parser that handles malformed data gracefully.

