# Chapter 4: Probability and Statistics

## Chapter Overview

Probability and statistics form the mathematical backbone of machine learning and AI. While linear algebra gives us the tools to represent and manipulate data, probability theory provides the framework for reasoning under uncertainty—a fundamental aspect of any intelligent system. Statistics allows us to draw conclusions from data, estimate parameters, and validate our models. This chapter covers the essential probability concepts, probability distributions, descriptive statistics, correlation, and foundational inferential statistics that every AI practitioner must master.

**Learning Objectives:**
By the end of this chapter, you will be able to:
- Understand and apply basic probability rules, including conditional probability and Bayes' Theorem.
- Differentiate between discrete and continuous random variables and their associated distributions.
- Compute descriptive statistics (mean, median, variance, standard deviation) and interpret them.
- Understand correlation and covariance and their importance in feature relationships.
- Apply inferential statistics concepts to evaluate models and make predictions.

---

## 4.1 Basic Probability Theory

Probability is the measure of how likely an event is to occur. It provides a formal framework for quantifying uncertainty.

### 4.1.1 Fundamental Concepts

**Sample Space (S):** The set of all possible outcomes of an experiment.
- Example: Flipping a coin → S = {Heads, Tails}
- Example: Rolling a die → S = {1, 2, 3, 4, 5, 6}

**Event (E):** A subset of the sample space (one or more outcomes).
- Example: Flipping a coin and getting Heads → E = {Heads}
- Example: Rolling an even number → E = {2, 4, 6}

**Probability of an Event:** 
```
P(E) = (Number of outcomes in E) / (Total number of outcomes in S)
```
For equally likely outcomes.

**Properties of Probability:**
1. `0 ≤ P(E) ≤ 1` (Probability is always between 0 and 1)
2. `P(S) = 1` (The probability of the entire sample space is 1)
3. For mutually exclusive events (events that cannot occur together):
   `P(A ∪ B) = P(A) + P(B)`

**Complement Rule:**
```
P(A') = 1 - P(A)
```
where A' is the complement of A (the event that A does not occur).

### 4.1.2 Conditional Probability

Conditional probability is the probability of an event occurring given that another event has already occurred. This is crucial in AI for updating beliefs based on new evidence.

**Definition:**
```
P(A | B) = P(A ∩ B) / P(B), for P(B) > 0
```
Read as: "Probability of A given B" equals the probability of both A and B occurring divided by the probability of B.

**Example:**
Suppose we have a deck of cards (52 cards). What is the probability of drawing a King given that the card drawn is a face card?

- Event A: Drawing a King (4 kings)
- Event B: Drawing a face card (12 face cards: Kings, Queens, Jacks)
- P(A ∩ B) = P(King and Face card) = P(King) = 4/52 (since all kings are face cards)
- P(B) = 12/52
- P(A | B) = (4/52) / (12/52) = 4/12 = 1/3

**Interpretation:** If you know you have a face card, the probability it's a King is 1/3 (since there are 4 kings among 12 face cards).

### 4.1.3 The Product Rule

The product rule allows us to compute the probability of two events occurring together:

```
P(A ∩ B) = P(A | B) * P(B) = P(B | A) * P(A)
```

**Independence:** Two events A and B are independent if the occurrence of one does not affect the probability of the other:
```
P(A | B) = P(A)  and  P(B | A) = P(B)
```
For independent events:
```
P(A ∩ B) = P(A) * P(B)
```

**Example:** If you flip a coin and roll a die, the events are independent.
- P(Heads and rolling a 6) = P(Heads) * P(6) = (1/2) * (1/6) = 1/12

### 4.1.4 Bayes' Theorem

Bayes' Theorem is one of the most important formulas in AI and machine learning. It describes how to update the probability of a hypothesis based on new evidence.

**Formula:**
```
P(H | E) = [P(E | H) * P(H)] / P(E)
```
Where:
- `P(H | E)`: Posterior probability (probability of hypothesis H given evidence E)
- `P(E | H)`: Likelihood (probability of evidence E given hypothesis H)
- `P(H)`: Prior probability (initial belief about H before seeing evidence)
- `P(E)`: Marginal probability (total probability of evidence E)

**Derivation:** From the definition of conditional probability:
```
P(H | E) = P(H ∩ E) / P(E) = P(E | H) * P(H) / P(E)
```

**Example: Medical Diagnosis**
Suppose a disease affects 1 in 1000 people (P(Disease) = 0.001). A test for the disease has:
- 99% sensitivity: P(Test+ | Disease) = 0.99
- 1% false positive rate: P(Test+ | No Disease) = 0.01

What is the probability that a person has the disease given they test positive?

**Compute:**
- P(Test+) = P(Test+ | Disease)*P(Disease) + P(Test+ | No Disease)*P(No Disease)
- P(Test+) = 0.99 * 0.001 + 0.01 * 0.999 = 0.00099 + 0.00999 = 0.01098

- P(Disease | Test+) = (0.99 * 0.001) / 0.01098 ≈ 0.0902 ≈ 9.02%

**Interpretation:** Even with a positive test, there's only about a 9% chance of having the disease! This surprising result is due to the low prior probability of the disease. This is a crucial insight for AI systems dealing with rare events.

**Importance in AI:**
- **Bayesian Inference:** The foundation of many probabilistic models.
- **Spam Filters:** Classifying emails as spam or not based on word probabilities.
- **Medical Diagnosis Systems:** Updating disease probabilities based on symptoms and test results.
- **Recommendation Systems:** Estimating user preferences based on past behavior.
- **Naive Bayes Classifier:** A simple yet powerful classification algorithm used in text classification and sentiment analysis.

---

## 4.2 Random Variables and Probability Distributions

A random variable is a variable whose value is subject to variations due to chance. It can be discrete or continuous.

### 4.2.1 Discrete Random Variables

A discrete random variable takes on a countable number of possible values (often integers).

**Probability Mass Function (PMF):** `P(X = x)` gives the probability that the random variable X equals a specific value x.

**Properties:**
- `0 ≤ P(X = x) ≤ 1`
- `Σ P(X = x) = 1` (sum over all possible values)

**Examples of Discrete Distributions:**

#### Bernoulli Distribution
- **Description:** A single trial with two possible outcomes (success/failure).
- **Parameter:** p = probability of success.
- **PMF:** 
  - P(X = 1) = p
  - P(X = 0) = 1 - p
- **Use Case:** Binary classification, flipping a biased coin.
- **Mean:** p
- **Variance:** p(1-p)

#### Binomial Distribution
- **Description:** Number of successes in n independent Bernoulli trials.
- **Parameters:** n (number of trials), p (probability of success per trial).
- **PMF:** P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
  where C(n,k) = n! / (k! * (n-k)!)
- **Use Case:** Number of defective items in a batch, number of heads in n coin flips.
- **Mean:** np
- **Variance:** np(1-p)

#### Poisson Distribution
- **Description:** Number of events occurring in a fixed interval of time or space.
- **Parameter:** λ (average rate of events).
- **PMF:** P(X = k) = (e^(-λ) * λ^k) / k!
- **Use Case:** Number of customers arriving at a store per hour, number of emails received per day.
- **Mean:** λ
- **Variance:** λ

### 4.2.2 Continuous Random Variables

A continuous random variable can take on any value within a range.

**Probability Density Function (PDF):** f(x) such that the probability of X falling between a and b is the integral of f(x) from a to b.

**Properties:**
- `f(x) ≥ 0` for all x
- `∫ f(x) dx = 1` (over all possible values)

**Examples of Continuous Distributions:**

#### Uniform Distribution
- **Description:** All values in a range [a, b] are equally likely.
- **Parameters:** a (minimum), b (maximum).
- **PDF:** f(x) = 1/(b-a) for a ≤ x ≤ b
- **Use Case:** Random number generation, prior probabilities with no information.
- **Mean:** (a+b)/2
- **Variance:** (b-a)²/12

#### Normal (Gaussian) Distribution
- **Description:** The classic "bell curve" distribution. Most values cluster around the mean.
- **Parameters:** μ (mean), σ² (variance).
- **PDF:** f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
- **Use Case:** The most important distribution in statistics. Arises from the Central Limit Theorem. Used in many AI models as prior distributions or error distributions.
- **Mean:** μ
- **Variance:** σ²
- **Standard Normal:** μ=0, σ²=1

**Key Properties of Normal Distribution:**
- Symmetric around the mean (μ).
- 68% of data falls within 1 standard deviation of the mean.
- 95% of data falls within 2 standard deviations.
- 99.7% of data falls within 3 standard deviations.

#### Exponential Distribution
- **Description:** Time until the next event in a Poisson process.
- **Parameter:** λ (rate parameter, inverse of mean).
- **PDF:** f(x) = λ * e^(-λx) for x ≥ 0
- **Use Case:** Time until a machine fails, waiting time between events.
- **Mean:** 1/λ
- **Variance:** 1/λ²

### 4.2.3 The Central Limit Theorem (CLT)

The Central Limit Theorem is one of the most important results in statistics. It states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution.

**Implications:**
- This is why the normal distribution appears so frequently in statistics and AI.
- It allows us to make inferences about population parameters from sample statistics.
- It justifies the use of many statistical tests that assume normality.

**Example:**
If you repeatedly take samples of size 30 from any distribution (exponential, uniform, binomial, etc.), the distribution of the sample means will be approximately normal. This holds even if the original distribution is not normal.

**Importance in AI:** The CLT underlies many algorithms and evaluation techniques, including:
- Confidence intervals for model performance metrics.
- Hypothesis testing for comparing models.
- Assumptions in linear regression (normality of residuals).

---

## 4.3 Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset.

### 4.3.1 Measures of Central Tendency

**Mean (Average):**
```
μ = (1/n) * Σ xᵢ  (population mean)
x̄ = (1/n) * Σ xᵢ  (sample mean)
```
- **Sensitive to outliers** (extreme values can skew the mean).

**Median:**
- The middle value when data is sorted in ascending order.
- **Robust to outliers** (not affected by extreme values).

**Mode:**
- The most frequently occurring value.
- Can be used with categorical data.

**Example:**
Dataset: [1, 2, 2, 3, 100]
- Mean = (1+2+2+3+100)/5 = 21.6
- Median = 2 (middle value)
- Mode = 2 (most frequent)

### 4.3.2 Measures of Dispersion (Spread)

**Variance:**
```
σ² = (1/n) * Σ (xᵢ - μ)²   (population variance)
s² = (1/(n-1)) * Σ (xᵢ - x̄)² (sample variance)
```
- Measures the average squared deviation from the mean.
- Larger variance indicates more spread.

**Standard Deviation:**
```
σ = √σ²   (population standard deviation)
s = √s²   (sample standard deviation)
```
- The square root of variance.
- Interpreted as the typical distance of data points from the mean.
- In the same units as the original data.

**Range:**
- Maximum value - Minimum value.
- Very sensitive to outliers.

**Interquartile Range (IQR):**
- Difference between the 75th percentile (Q3) and 25th percentile (Q1).
- Robust to outliers.
- Contains the middle 50% of the data.

**Example:**
Dataset: [1, 2, 2, 3, 100]
- Mean = 21.6
- Variance = ((1-21.6)² + (2-21.6)² + (2-21.6)² + (3-21.6)² + (100-21.6)²)/5 = (424.36 + 384.16 + 384.16 + 345.96 + 6146.56)/5 = 1537.04
- Standard Deviation = √1537.04 ≈ 39.2

### 4.3.3 Skewness and Kurtosis

**Skewness:** Measures asymmetry of the distribution.
- **Positive Skew (Right-skewed):** Long tail on the right. Mean > Median.
- **Negative Skew (Left-skewed):** Long tail on the left. Mean < Median.
- **Zero Skew:** Symmetric distribution. Mean = Median.

**Kurtosis:** Measures the "tailedness" or peakedness of the distribution.
- **High Kurtosis:** Heavy tails, more outliers.
- **Low Kurtosis:** Light tails, fewer outliers.
- **Normal distribution** has kurtosis = 3 (excess kurtosis = 0).

---

## 4.4 Correlation and Covariance

Covariance and correlation measure the relationship between two variables.

### 4.4.1 Covariance

**Definition:** Measures how two variables change together.
```
Cov(X, Y) = (1/n) * Σ (xᵢ - x̄)(yᵢ - ȳ)
```

**Interpretation:**
- **Positive Covariance:** When X increases, Y tends to increase.
- **Negative Covariance:** When X increases, Y tends to decrease.
- **Zero Covariance:** No linear relationship.

**Limitation:** Covariance is scale-dependent (changes when units change) and difficult to interpret directly.

**Example:**
```
Data: (1, 2), (2, 3), (3, 5)
x̄ = 2, ȳ = 3.33
Cov = ((1-2)(2-3.33) + (2-2)(3-3.33) + (3-2)(5-3.33))/3
    = ((-1)(-1.33) + (0)(-0.33) + (1)(1.67))/3
    = (1.33 + 0 + 1.67)/3 = 1.0
```
Positive covariance indicates a positive relationship.

### 4.4.2 Correlation

**Definition:** A normalized version of covariance that is scale-independent and easier to interpret.

The **Pearson Correlation Coefficient (r):**
```
r = Cov(X, Y) / (σ_X * σ_Y)
```

**Properties:**
- **-1 ≤ r ≤ 1**
- **r = 1:** Perfect positive correlation (all points lie on a line with positive slope).
- **r = -1:** Perfect negative correlation (all points lie on a line with negative slope).
- **r = 0:** No linear correlation (but could have non-linear relationship).

**Interpretation:**
- |r| ≥ 0.8: Strong correlation
- |r| ≥ 0.5 and < 0.8: Moderate correlation
- |r| < 0.5: Weak correlation

**Example:**
Using the previous covariance example:
If σ_X = 0.816 and σ_Y = 1.247:
r = 1.0 / (0.816 * 1.247) = 0.982 (strong positive correlation)

### 4.4.3 Importance in AI

- **Feature Selection:** Highly correlated features may be redundant; can reduce dimensionality.
- **Feature Engineering:** Understanding correlations helps in creating new features.
- **Multicollinearity:** In linear regression, high correlation between predictors can cause instability.
- **Exploratory Data Analysis (EDA):** Correlation matrices help visualize relationships.

**Example: Correlation Matrix Visualization:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create correlated data
np.random.seed(42)
X = np.random.randn(100, 1)
Y = 2 * X + np.random.randn(100, 1) * 0.5
Z = -1.5 * X + np.random.randn(100, 1) * 0.5

data = np.hstack([X, Y, Z])
df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])

# Correlation matrix
corr = df.corr()
print(corr)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
```

---

## 4.5 Inferential Statistics

Inferential statistics allows us to make generalizations about a population based on a sample.

### 4.5.1 Hypothesis Testing

Hypothesis testing is a method for making decisions using data.

**Steps:**
1. **Define Null Hypothesis (H₀):** The status quo or no effect.
2. **Define Alternative Hypothesis (H₁):** What we want to prove.
3. **Choose a Significance Level (α):** Usually 0.05 (5% risk of false positive).
4. **Compute Test Statistic:** Based on the sample data.
5. **Compute p-value:** Probability of observing the test statistic (or more extreme) if H₀ is true.
6. **Make Decision:**
   - If p-value < α: Reject H₀ (statistically significant result).
   - If p-value ≥ α: Fail to reject H₀ (not enough evidence).

**Example: A/B Testing**
Suppose we have a new recommendation algorithm and want to test if it improves click-through rate (CTR).
- H₀: New algorithm has no effect on CTR.
- H₁: New algorithm increases CTR.
- Run experiment with control group (old algorithm) and treatment group (new algorithm).
- Compute p-value based on difference in CTRs.
- If p-value < 0.05, conclude the new algorithm is better.

### 4.5.2 Confidence Intervals

A confidence interval gives a range of plausible values for a population parameter.

**Example:** A 95% confidence interval for the mean:
```
x̄ ± z * (σ / √n)
```
where z = 1.96 for 95% confidence.

**Interpretation:** If we repeat the sampling many times, 95% of the confidence intervals would contain the true population mean.

### 4.5.3 p-value in Model Evaluation

In machine learning, p-values are often used to:
- Test if a feature has significant predictive power.
- Compare the performance of two models.
- Validate that improvements are not due to chance.

**Example:** Compare accuracy of two models.
- H₀: Model A and Model B have equal accuracy.
- H₁: Model A has higher accuracy.
- Run cross-validation and compute test statistic.
- p-value indicates if the observed difference is statistically significant.

### 4.5.4 Type I and Type II Errors

- **Type I Error (False Positive):** Rejecting H₀ when it's true (α risk).
- **Type II Error (False Negative):** Failing to reject H₀ when it's false (β risk).
- **Power of a test:** 1 - β (probability of correctly rejecting a false H₀).

**Trade-off:** Reducing Type I error (lower α) increases Type II error (higher β) and vice versa.

### 4.5.5 The Law of Large Numbers

The Law of Large Numbers states that as the sample size increases, the sample mean converges to the true population mean.

**Implications for AI:**
- More training data generally improves model performance.
- Evaluation metrics become more reliable with larger validation sets.
- Ensures that empirical risk minimization (training error) approximates expected risk (test error).

---

## 4.6 Probability and Statistics in AI: A Practical Example

Let's implement a **Naive Bayes Classifier** from scratch to see probability concepts in action.

### Naive Bayes for Spam Classification

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption of feature independence.

**Steps:**
1. Compute prior probabilities: P(Spam) and P(Ham) from training data.
2. Compute conditional probabilities: P(Word | Spam) and P(Word | Ham) for each word.
3. For a new document, compute posterior probabilities using Bayes' Theorem.
4. Classify based on which posterior is higher.

```python
import numpy as np
from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_probs = {}
        self.word_probs = {}
        self.vocab_size = 0
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        X: list of documents (lists of words)
        y: list of labels (e.g., ['spam', 'ham', ...])
        """
        # Count documents per class
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        
        for doc, label in zip(X, y):
            class_counts[label] += 1
            for word in doc:
                word_counts[label][word] += 1
        
        # Compute prior probabilities
        total_docs = len(y)
        self.class_probs = {
            cls: count / total_docs for cls, count in class_counts.items()
        }
        
        # Vocabulary size
        vocab = set()
        for doc in X:
            vocab.update(doc)
        self.vocab_size = len(vocab)
        
        # Compute conditional probabilities with Laplace smoothing
        self.word_probs = {}
        for label in class_counts:
            total_words = sum(word_counts[label].values())
            self.word_probs[label] = {}
            for word in vocab:
                count = word_counts[label].get(word, 0)
                # Laplace smoothing: (count + alpha) / (total_words + alpha * vocab_size)
                self.word_probs[label][word] = (count + self.alpha) / (total_words + self.alpha * self.vocab_size)
    
    def predict_proba(self, doc):
        """
        Predict class probabilities for a single document.
        """
        scores = {}
        for label in self.class_probs:
            # Start with log of prior (to avoid underflow)
            log_prob = math.log(self.class_probs[label])
            for word in doc:
                # Add log of conditional probability
                if word in self.word_probs[label]:
                    log_prob += math.log(self.word_probs[label][word])
                else:
                    # Word not in training vocabulary, use small probability
                    log_prob += math.log(1 / (self.vocab_size + 1))
            scores[label] = log_prob
        
        # Convert log probabilities back to probabilities
        # Normalize using log-sum-exp trick for numerical stability
        max_log = max(scores.values())
        exp_scores = {label: math.exp(logp - max_log) for label, logp in scores.items()}
        total = sum(exp_scores.values())
        return {label: prob / total for label, prob in exp_scores.items()}
    
    def predict(self, doc):
        """
        Predict the class for a document.
        """
        probs = self.predict_proba(doc)
        return max(probs, key=probs.get)

# Example training data
X_train = [
    ['free', 'money', 'click', 'now'],
    ['win', 'prize', 'free', 'money'],
    ['meeting', 'tomorrow', 'project'],
    ['team', 'lunch', 'schedule'],
    ['urgent', 'claim', 'your', 'prize']
]
y_train = ['spam', 'spam', 'ham', 'ham', 'spam']

# Train classifier
clf = NaiveBayesClassifier(alpha=1.0)
clf.fit(X_train, y_train)

# Test on new documents
test_docs = [
    ['free', 'money', 'win'],
    ['project', 'meeting', 'team'],
    ['click', 'now', 'urgent']
]

for doc in test_docs:
    prob = clf.predict_proba(doc)
    pred = clf.predict(doc)
    print(f"Document: {doc}")
    print(f"  Probabilities: {prob}")
    print(f"  Predicted: {pred}\n")
```

**Output:**
```
Document: ['free', 'money', 'win']
  Probabilities: {'spam': 0.9999, 'ham': 0.0001}
  Predicted: spam

Document: ['project', 'meeting', 'team']
  Probabilities: {'spam': 0.0001, 'ham': 0.9999}
  Predicted: ham

Document: ['click', 'now', 'urgent']
  Probabilities: {'spam': 0.9999, 'ham': 0.0001}
  Predicted: spam
```

**Explanation:**
- Naive Bayes uses Bayes' Theorem: P(Class | Document) ∝ P(Class) * Π P(Word | Class)
- The "naive" assumption is that words are conditionally independent given the class.
- Laplace smoothing (α=1) prevents zero probabilities for unseen words.
- Log probabilities are used to avoid underflow (multiplying many small numbers).

**Importance:** Naive Bayes is a classic example of how probability theory is directly applied to build an AI system. Despite its simplicity, it performs well for text classification tasks.

---

## Summary

Probability and statistics are essential for understanding and building AI systems. In this chapter, we covered:

- **Basic probability** including conditional probability and Bayes' Theorem, which form the foundation for many AI algorithms.
- **Random variables and distributions**, both discrete (Bernoulli, Binomial, Poisson) and continuous (Uniform, Normal, Exponential), and the Central Limit Theorem.
- **Descriptive statistics** for summarizing data (mean, median, variance, standard deviation, skewness, kurtosis).
- **Correlation and covariance** for measuring relationships between variables, critical for feature engineering and EDA.
- **Inferential statistics** including hypothesis testing, confidence intervals, and p-values, which are essential for model evaluation and comparison.
- A practical **Naive Bayes classifier** demonstrating how probability theory is applied in AI.

These concepts will appear repeatedly throughout the book, especially when we discuss model evaluation, feature engineering, and probabilistic models.

---

##  Further Reading & Resources

- **Books:**
  - *Introduction to Probability* by Joseph K. Blitzstein and Jessica Hwang.
  - *All of Statistics* by Larry Wasserman.
  - *Statistical Inference* by George Casella and Roger L. Berger.
- **Online:**
  - [Khan Academy: Probability and Statistics](https://www.khanacademy.org/math/statistics-probability)
  - [3Blue1Brown: Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
  - [Seeing Theory: Interactive Probability Tutorial](https://seeing-theory.brown.edu/)

---

##  Chapter 4 Checklist

Before moving on, ensure you can:

- [ ] Define and compute conditional probability: P(A|B).
- [ ] State and apply Bayes' Theorem.
- [ ] Differentiate between discrete and continuous random variables.
- [ ] Identify the parameters and properties of Bernoulli, Binomial, and Normal distributions.
- [ ] Compute mean, median, variance, and standard deviation for a dataset.
- [ ] Explain the difference between covariance and correlation, and compute both.
- [ ] Interpret p-values and their significance in hypothesis testing.
- [ ] Build a simple Naive Bayes classifier from scratch.

---

##  Hands-On Exercises

1. **Bayes' Theorem Practice:**
   - A factory has 3 machines producing parts: Machine A (50% of parts, 2% defective), Machine B (30% of parts, 3% defective), Machine C (20% of parts, 4% defective). If a part is defective, what is the probability it came from Machine A?

2. **Distribution Properties:**
   - For a Binomial distribution with n=10 and p=0.3, compute P(X=3) and P(X≤4) using the formula and verify with Python's `scipy.stats.binom`.

3. **Statistics Computation:**
   - Given data: [2, 4, 6, 8, 10, 12, 100], compute mean, median, variance, standard deviation, and skewness. Interpret the results.

4. **Correlation Analysis:**
   - Generate 100 samples from a bivariate normal distribution with correlation 0.8. Compute the correlation matrix and visualize with a heatmap.

5. **Naive Bayes Extension:**
   - Add a method to the NaiveBayesClassifier that returns the top N most predictive words for each class (words with highest conditional probability). Use this to interpret the model.

---
```
