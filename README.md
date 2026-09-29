# Employee-salary-predictor-
# 📊 Employee Salary Predictor (Multiple Linear Regression)

## 📌 Project Overview
This project demonstrates the application of **Multiple Linear Regression** to model corporate compensation patterns. It features an end-to-end Machine Learning pipeline that simulates a human resources dataset, preprocesses features, trains a predictive model, and visualizes model accuracy.

---

## 🛠️ Step-by-Step Architectural Pipeline

### Step 1: Synthesizing the Data Architecture
The script generates a realistic dataset of **200 employee profiles** using mathematical distributions.
* **Independent Features (X):** `YearsExperience` (continuous range), `JobTier` (1=Junior, 2=Mid, 3=Senior), and `EducationLevel` (0=Bachelors, 1=Masters, 2=PhD).
* **Dependent Target (y):** `Salary`. Formulated with a baseline of \$40,000, multi-variable increments, and a random Gaussian noise factor to simulate real-world corporate anomalies.

### Step 2: Data Splitting Protocol
To ensure strict validation and prevent data leakage, the dataset is systematically partitioned:
* **80% Training Set:** Utilized by the algorithm to discover the mathematical boundaries and weights.
* **20% Testing Set:** Completely isolated from the model, serving as a blind verification layer to test the final predictive accuracy.

### Step 3: Model Execution & Training
We initialize a `LinearRegression()` object from the **scikit-learn** library. By calling `.fit(X_train, y_train)`, the system runs an ordinary least squares optimization to solve the linear system:

\[\text{Salary} = \beta_0 + \beta_1(\text{Experience}) + \beta_2(\text{JobTier}) + \beta_3(\text{EducationLevel}) + \epsilon\]

### Step 4: Metric Evaluation Matrix
Once the model produces predictions on the unseen test array, it evaluates performance using three industry-standard core metrics:
* **Mean Absolute Error (MAE):** Tells us the average dollar amount the model's predictions miss by.
* **Root Mean Squared Error (RMSE):** Measures prediction variance, penalizing larger stray errors more heavily.
* **R-squared (R²) Score:** Quantifies how much variance our model accounts for. A score of **~0.94** demonstrates that our selected features accurately explain **94%** of salary variances.

### Step 5: Inference Engine (Custom Predictions)
A functional script layer allows users to feed arbitrary values into the trained model. By providing custom features (e.g., 5.5 years of experience, Tier 2 role, Master's degree), the model instantly calculates a tailored market salary estimate.

### Step 6: Predictive Data Visualization
Using **matplotlib**, the model plots actual salaries against predicted outputs in a scatter graph. A dashed **y = x reference line** represents mathematical perfection. The proximity of the generated data points to this line visually confirms the model's high precision and low variance.

---

## 🧰 Tech Stack & Libraries Used
- **Python** (Core Language)
- **Pandas** (Data Manipulation)
- **NumPy** (Mathematical Computations)
- **Scikit-Learn** (Machine Learning Model & Metrics)
- **Matplotlib** (Data Visualization)
