

# Chapter 3: Linear Algebra for AI

## Chapter Overview

Linear algebra is the mathematical foundation upon which most of modern AI and machine learning is built. It provides the language and tools to represent, manipulate, and transform data in high-dimensional spaces. Whether you're working with images, text, or tabular data, you're essentially dealing with vectors and matrices. This chapter will take you from the basics of algebra and functions through to the essential linear algebra concepts that you will use daily as an AI practitioner. We'll cover vectors, matrices, operations, and eigenvalues/eigenvectors—all with a focus on their practical applications in AI.

**Learning Objectives:**
By the end of this chapter, you will be able to:
- Understand the concept of vectors and perform vector operations.
- Represent data and transformations using matrices.
- Perform matrix operations including addition, multiplication, and transposition.
- Solve systems of linear equations.
- Explain the significance of eigenvalues and eigenvectors in AI applications.
- Implement basic linear algebra operations using NumPy.

---

## 3.1 Prerequisite: Algebra and Functions

Before diving into linear algebra, let's briefly review the foundational concepts of algebra and functions, which form the basis of everything we'll discuss.

### 3.1.1 Basic Algebra

Algebra deals with mathematical symbols and the rules for manipulating them. It allows us to express relationships between quantities using variables.

**Key Concepts:**
- **Variables:** Symbols (like `x`, `y`, `z`) that represent unknown or changeable values.
- **Constants:** Fixed values (like 3, π, e).
- **Expressions:** Combinations of variables, constants, and operations (e.g., `3x + 2`).
- **Equations:** Statements that two expressions are equal (e.g., `3x + 2 = 8`).
- **Solving Equations:** Finding the values of variables that make the equation true.

**Example:** Solve for `x` in `2x + 5 = 13`
```
2x + 5 = 13
2x = 13 - 5
2x = 8
x = 4
```

### 3.1.2 Functions

A function is a relation that maps each input value to exactly one output value. It's a fundamental concept in both mathematics and AI.

**Notation:** `f(x) = y`, where `x` is the input (domain), `y` is the output (range), and `f` is the function rule.

**Common Functions in AI:**
- **Linear:** `f(x) = mx + b` (straight line)
- **Polynomial:** `f(x) = ax² + bx + c` (parabola)
- **Exponential:** `f(x) = eˣ` (rapid growth/decay)
- **Logistic (Sigmoid):** `f(x) = 1 / (1 + e⁻ˣ)` (bounded between 0 and 1, used in classification)

**Why Functions Matter in AI:** Every machine learning model is essentially a function that maps inputs to outputs. Neural networks are complex compositions of many simple functions. Understanding functions is crucial for:
- Forward propagation (computing predictions).
- Backpropagation (computing gradients).
- Activation functions (introducing non-linearity).

---

## 3.2 Vectors: The Building Blocks of Data

### 3.2.1 What is a Vector?

In the context of AI and machine learning, a vector is an ordered list of numbers. Think of it as a point in space with a direction and magnitude.

**Representation:**
A vector with `n` elements (an n-dimensional vector) is written as:
```
v = [v₁, v₂, ..., vₙ]
```

**Geometric Interpretation:**
- **2D Vector:** `v = [3, 4]` represents a point in a 2D plane (3 units right, 4 units up).
- **3D Vector:** `v = [1, 2, 3]` represents a point in 3D space.

**Visualization:**

```
      y
      ^
      |
     4|   * (3,4)
      |
     2|
      |
      +-------> x
          3
```

### 3.2.2 Vector Operations

#### Addition and Subtraction
- **Addition:** Add corresponding elements.
  - `u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]`
  - **Example:** `[1, 2] + [3, 4] = [4, 6]`
- **Subtraction:** Subtract corresponding elements.
  - `u - v = [u₁ - v₁, u₂ - v₂, ..., uₙ - vₙ]`
  - **Example:** `[5, 7] - [2, 3] = [3, 4]`

**Geometric Interpretation:** Vector addition corresponds to placing the tail of one vector at the head of the other (parallelogram law).

#### Scalar Multiplication
Multiplying a vector by a scalar (a single number) scales the vector.
- `c * v = [c * v₁, c * v₂, ..., c * vₙ]`
- **Example:** `3 * [1, 2, 3] = [3, 6, 9]`
- **Geometric Interpretation:** Changes the magnitude (length) of the vector. If `c > 1`, it stretches; if `0 < c < 1`, it shrinks; if `c < 0`, it flips direction.

#### Dot Product (Scalar Product)
The dot product takes two vectors and returns a single number (a scalar). It's central to many AI operations, including similarity calculations and linear transformations.

**Definition:**
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σᵢ uᵢvᵢ
```

**Example:**
```
u = [1, 2, 3]
v = [4, 5, 6]
u · v = (1 * 4) + (2 * 5) + (3 * 6) = 4 + 10 + 18 = 32
```

**Geometric Interpretation:** The dot product is related to the angle `θ` between two vectors:
```
u · v = ||u|| * ||v|| * cos(θ)
```
where `||u||` is the magnitude (length) of vector `u`.

This means:
- If `u · v = 0`, the vectors are orthogonal (perpendicular).
- If `u · v > 0`, the vectors point in a similar direction.
- If `u · v < 0`, the vectors point in opposite directions.

**Importance in AI:** Dot products are used in:
- **Weighted sums:** The core operation in neural networks (`output = weights · inputs + bias`).
- **Similarity:** Cosine similarity (used in recommendation systems and information retrieval).
- **Attention mechanisms:** Computing how much focus to give to different parts of the input.

#### Norm (Magnitude)
The norm of a vector is its length or magnitude. The most common is the Euclidean norm (L2 norm):
```
||v|| = √(v₁² + v₂² + ... + vₙ²) = √(v · v)
```

**Example:** For `v = [3, 4]`, `||v|| = √(9 + 16) = √25 = 5`

**Importance in AI:** Norms are used for:
- **Normalization:** Scaling input data.
- **Regularization:** Penalizing large weights to prevent overfitting.

---

## 3.3 Matrices: Representing and Transforming Data

### 3.3.1 What is a Matrix?

A matrix is a rectangular array of numbers arranged in rows and columns. It's a powerful way to represent multiple vectors or linear transformations.

**Representation:**
A matrix with `m` rows and `n` columns (an m×n matrix) is written as:

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [ ...  ...  ...  ... ]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

**Example:**
```
A = [1  2  3]
    [4  5  6]        (2×3 matrix)
```

In Python (NumPy), we represent matrices as 2D arrays:
```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
```

### 3.3.2 Matrix Operations

#### Addition and Subtraction
- **Requirements:** Matrices must have the same dimensions.
- **Operation:** Add or subtract corresponding elements.

**Example:**
```
A = [1  2]    B = [5  6]    A + B = [6  8]
    [3  4]        [7  8]            [10 12]
```

#### Scalar Multiplication
Multiply each element of the matrix by a scalar.

**Example:**
```
A = [1  2]   2 * A = [2  4]
    [3  4]           [6  8]
```

#### Matrix Multiplication
Matrix multiplication is one of the most important operations in AI. It's not element-wise; instead, it's defined as the dot product of rows and columns.

**Requirements:** For `A (m×n)` and `B (n×p)`, the number of columns in `A` must equal the number of rows in `B`. The result is an `m×p` matrix.

**Definition:** The element `(i, j)` of the product `C = A * B` is:
```
Cᵢⱼ = (row i of A) · (column j of B) = Σₖ Aᵢₖ * Bₖⱼ
```

**Example:**
```
A = [1  2]    B = [5  6]    A * B = [1*5 + 2*7    1*6 + 2*8] = [19  22]
    [3  4]        [7  8]            [3*5 + 4*7    3*6 + 4*8]   [43  50]
```

**Properties:**
- **Not Commutative:** `A * B ≠ B * A` (in general).
- **Associative:** `(A * B) * C = A * (B * C)`.
- **Distributive:** `A * (B + C) = A * B + A * C`.

**Importance in AI:** Matrix multiplication is the backbone of neural networks. A single layer's forward pass is: `output = activation(W * input + b)`, where `W` is the weight matrix, `input` is the input vector (or batch of inputs), and `b` is the bias vector.

#### Transpose
The transpose of a matrix `A` (denoted as `Aᵀ`) is obtained by swapping rows with columns.

**Definition:** The element at position `(i, j)` in `A` becomes the element at position `(j, i)` in `Aᵀ`.

**Example:**
```
A = [1  2  3]    Aᵀ = [1  4]
    [4  5  6]          [2  5]
                       [3  6]
```

**Properties:**
- `(Aᵀ)ᵀ = A`
- `(A * B)ᵀ = Bᵀ * Aᵀ`

**Importance in AI:** Transposes are used in:
- Changing the orientation of data.
- Computing gradients in backpropagation.
- Dealing with inner and outer products.

---

## 3.4 Linear Algebra Concepts

### 3.4.1 Solving Systems of Linear Equations

A system of linear equations is a set of equations that can be written in the form:
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ
```

This can be compactly represented as the matrix equation:
```
A * x = b
```
where `A` is the coefficient matrix, `x` is the vector of unknowns, and `b` is the vector of constants.

**Example:**
```
2x + 3y = 8
4x + y = 6

A = [2  3]    x = [x]    b = [8]
    [4  1]        [y]        [6]
```

**Solutions:**
- **Unique solution:** If `det(A) ≠ 0` (matrix is invertible).
- **No solution:** Inconsistent equations.
- **Infinite solutions:** Underdetermined system.

**Solving in Python (NumPy):**
```python
import numpy as np

A = np.array([[2, 3], [4, 1]])
b = np.array([8, 6])

# Solution: x = A⁻¹ * b
x = np.linalg.solve(A, b)
print(x)  # Output: [1.  2.]  (x=1, y=2)
```

**Importance in AI:**
- **Linear regression:** Fitting a line to data involves solving a system of equations.
- **Principal Component Analysis (PCA):** Involves solving eigenvalue equations.
- **Optimization:** Many algorithms solve linear systems as a sub-step.

### 3.4.2 Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in linear algebra that appear in many AI algorithms.

**Definition:** Given a square matrix `A`, a vector `v` is an **eigenvector** of `A` if:
```
A * v = λ * v
```
where `λ` is a scalar called the **eigenvalue** associated with `v`.

**Interpretation:**
- Multiplying matrix `A` by its eigenvector `v` only scales `v` by a factor `λ`.
- The direction of `v` remains unchanged (or can be reversed if `λ` is negative).

**Computing Eigenvalues:**
Eigenvalues are found by solving the characteristic equation:
```
det(A - λI) = 0
```
where `I` is the identity matrix.

**Example:**
```
A = [2  1]
    [1  2]

Eigenvalues: λ₁ = 3, λ₂ = 1
Eigenvectors: v₁ = [1, 1], v₂ = [-1, 1]
```

**Checking:**
```
A * v₁ = [2  1] * [1] = [3] = 3 * [1]
         [1  2]   [1]   [3]     [1]

A * v₂ = [2  1] * [-1] = [-1] = 1 * [-1]
         [1  2]   [ 1]   [ 1]     [ 1]
```

**Computation in Python:**
```python
import numpy as np

A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)    # [3.  1.]
print(eigenvectors)   # [[0.70710678 -0.70710678]
                      #  [0.70710678  0.70710678]]
```

**Importance in AI:**
- **Principal Component Analysis (PCA):** Eigenvectors of the covariance matrix represent the principal components (directions of maximum variance).
- **Singular Value Decomposition (SVD):** A generalization of eigenvalue decomposition for non-square matrices, used in dimensionality reduction, recommendation systems, and matrix factorization.
- **Stability and convergence:** In optimization, the eigenvalues of the Hessian matrix (second derivatives) determine the curvature and convergence properties of gradient descent.
- **PageRank:** The PageRank algorithm used by Google is essentially an eigenvector computation.

---

## 3.5 Practical Implementation with NumPy

Let's solidify our understanding with hands-on examples using NumPy.

```python
import numpy as np

# --- 3.5.1 Creating Vectors and Matrices ---
# Vector (1D array)
v = np.array([1, 2, 3, 4])
print("Vector v:", v)
print("Shape:", v.shape)         # (4,)

# Matrix (2D array)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nMatrix A:\n", A)
print("Shape:", A.shape)         # (3, 3)

# --- 3.5.2 Vector Operations ---
u = np.array([1, 2, 3])
w = np.array([4, 5, 6])

# Addition
print("\nVector addition:", u + w)          # [5 7 9]

# Scalar multiplication
print("Scalar multiplication:", 2 * u)      # [2 4 6]

# Dot product
print("Dot product:", np.dot(u, w))         # 32
print("Alternative:", u @ w)                # 32 (Python 3.5+)

# Norm
print("L2 norm:", np.linalg.norm(u))        # 3.741657...

# --- 3.5.3 Matrix Operations ---
B = np.array([[1, 2], [3, 4], [5, 6]])     # 3×2 matrix
C = np.array([[7, 8], [9, 10]])             # 2×2 matrix

# Matrix multiplication
# A (3×3) * B (3×2) -> Error! Dimensions don't match for standard multiplication
# B (3×2) * C (2×2) -> Result is 3×2
result = B @ C
print("\nMatrix multiplication (B @ C):\n", result)
# [[1*7+2*9  1*8+2*10]    => [[25 28]
#  [3*7+4*9  3*8+4*10]       [57 64]
#  [5*7+6*9  5*8+6*10]]      [89 100]]

# Transpose
print("\nTranspose of B:\n", B.T)
# [[1 3 5]
#  [2 4 6]]

# --- 3.5.4 Solving Linear Systems ---
A_solve = np.array([[2, 3], [4, 1]])
b_solve = np.array([8, 6])
x = np.linalg.solve(A_solve, b_solve)
print("\nSolution to A*x = b:", x)           # [1. 2.]

# --- 3.5.5 Eigenvalues and Eigenvectors ---
A_eig = np.array([[2, 1], [1, 2]])
eigenvals, eigenvecs = np.linalg.eig(A_eig)
print("\nEigenvalues:", eigenvals)           # [3. 1.]
print("Eigenvectors (columns):\n", eigenvecs)
# [[0.70710678 -0.70710678]
#  [0.70710678  0.70710678]]

# Verify: A * v = λ * v
v1 = eigenvecs[:, 0]  # First eigenvector
lambda1 = eigenvals[0]
print("\nVerification for first eigenpair:")
print("A * v1 =", A_eig @ v1)
print("λ * v1 =", lambda1 * v1)
```

---

## 3.6 Linear Algebra in AI: A Practical Example

Let's see how linear algebra powers a core AI algorithm: **Principal Component Analysis (PCA)**.

PCA is a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible. It's widely used for data compression, visualization, and noise reduction.

### The Steps:

1. **Center the data:** Subtract the mean from each feature.
2. **Compute the covariance matrix:** `C = (1/(n-1)) * Xᵀ * X`, where `X` is the centered data matrix.
3. **Compute eigenvectors and eigenvalues** of the covariance matrix.
4. **Select the top `k` eigenvectors** corresponding to the largest eigenvalues (principal components).
5. **Project the data** onto the principal components: `X_projected = X * W`, where `W` is the matrix of top eigenvectors.

### Python Example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic 2D data
np.random.seed(42)
mean = [0, 0]
cov = [[2, 1.5], [1.5, 2]]
X = np.random.multivariate_normal(mean, cov, 100)

# --- Step 1: Center the data ---
X_centered = X - np.mean(X, axis=0)

# --- Step 2: Compute covariance matrix ---
cov_matrix = (1 / (X.shape[0] - 1)) * (X_centered.T @ X_centered)

# --- Step 3: Compute eigenvalues and eigenvectors ---
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

# --- Step 4: Sort eigenvalues and corresponding eigenvectors ---
idx = np.argsort(eigenvals)[::-1]  # descending order
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# --- Step 5: Project data onto first principal component (1D) ---
pc1 = eigenvecs[:, 0].reshape(-1, 1)
X_pca = X_centered @ pc1

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original data')

# Plot the first principal component (direction)
arrow_length = 2
plt.arrow(0, 0, pc1[0]*arrow_length, pc1[1]*arrow_length,
          head_width=0.2, head_length=0.2, fc='red', ec='red', label='PC1')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA on 2D Data')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"Eigenvalues: {eigenvals}")
print(f"First principal component: {pc1.flatten()}")
```

**Output:**
```
Eigenvalues: [3.454 0.502]
First principal component: [0.719 0.695]
```

**Interpretation:**
- The first principal component captures ~87% of the variance (3.454 / (3.454+0.502)).
- The data can be projected onto this one dimension with minimal information loss.

---

## Summary

Linear algebra is the language of data in AI. In this chapter, we've covered:

- **Vectors** as the fundamental unit of data, including operations like addition, scalar multiplication, and the dot product.
- **Matrices** for representing multiple vectors and linear transformations, including matrix multiplication and transposition.
- **Solving systems of linear equations** and understanding their geometric interpretations.
- **Eigenvalues and eigenvectors**, their computation, and their critical role in algorithms like PCA.

With NumPy, you now have a practical tool to perform these operations efficiently. This foundation will be essential as we move to probability, statistics, and ultimately the implementation of machine learning algorithms.

---

##  Further Reading & Resources

- **Books:**
  - *Linear Algebra Done Right* by Sheldon Axler.
  - *Introduction to Linear Algebra* by Gilbert Strang.
- **Online:**
  - [3Blue1Brown's Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) – Visual and intuitive explanations.
  - [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- **Cheat Sheet:**
  - [NumPy Linear Algebra Reference](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)

---

##  Chapter 3 Checklist

Before moving on, ensure you can:

- [ ] Define a vector and perform addition, scalar multiplication, and dot product operations.
- [ ] Compute the norm of a vector and understand its geometric meaning.
- [ ] Define a matrix and perform addition, scalar multiplication, and multiplication.
- [ ] Compute the transpose of a matrix and explain its use.
- [ ] Solve a system of linear equations using `np.linalg.solve`.
- [ ] Explain what eigenvalues and eigenvectors are and how they are computed in NumPy.
- [ ] Describe at least one AI application that uses eigenvalues and eigenvectors.

---

##  Hands-On Exercises

1. **Vector Operations:**
   - Create two 4-dimensional vectors `u = [2, -1, 0, 3]` and `v = [1, 4, -2, 1]`.
   - Compute `u + v`, `2*u`, `u - v`, the dot product `u·v`, and the norm of `u`.
   - Verify that `norm(u) = sqrt(u·u)`.

2. **Matrix Multiplication:**
   - Given `A = [[1, 2], [3, 4]]` and `B = [[5, 6], [7, 8]]`, compute `A*B` and `B*A` by hand and verify with NumPy.
   - Explain why `A*B ≠ B*A` in this case.

3. **Eigenvalues and Eigenvectors:**
   - Compute eigenvalues and eigenvectors for `A = [[4, 1], [2, 3]]` using NumPy.
   - Verify that `A * v = λ * v` for one eigenpair.

4. **Solving Linear Systems:**
   - Solve the system: `3x + 2y = 7`, `x - y = 1`.
   - Interpret the solution geometrically (intersection of two lines).

5. **PCA Implementation:**
   - Modify the PCA example to reduce data from 3D to 2D using synthetic data.
   - Visualize the original 3D data and the projected 2D data.
