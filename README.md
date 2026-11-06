# Medical Insurance Cost Prediction using Machine Learning

## Project Overview

This project focuses on analyzing medical insurance data and developing a Machine Learning model to predict individual medical insurance charges based on personal and lifestyle-related factors.

Using Python, statistical analysis and Scikit-learn, I explored how age, BMI, smoking habits, number of children and geographical region influence insurance costs.

The project goes beyond basic Exploratory Data Analysis by incorporating hypothesis testing, ANOVA, correlation analysis, multicollinearity checks, feature selection and feature engineering to improve model performance.

## Project Objectives

- Understand the factors influencing medical insurance charges.
- Perform data cleaning and exploratory data analysis.
- Identify outliers and analyze their relationship with smoking habits.
- Apply statistical tests to examine relationships between variables.
- Identify important predictors using correlation and regression analysis.
- Check multicollinearity using VIF.
- Perform feature selection using Backward Elimination.
- Build and evaluate a Multiple Linear Regression model.
- Improve model performance through feature engineering.
- Develop a custom function to predict insurance charges for new individuals.

## Dataset Information

- **Dataset:** Medical Insurance Dataset
- **Initial Records:** 1,338
- **Records after duplicate removal:** 1,337
- **Features:** 7
- **Target Variable:** Charges

### Dataset Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Individual medical insurance charges |

## Tools & Technologies

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical hypothesis testing |
| Statsmodels | OLS regression and ANOVA |
| Scikit-learn | Machine Learning and evaluation |
| Jupyter Notebook | Development environment |

## Project Workflow

### 1. Data Exploration

Performed initial dataset inspection using:

- `head()`
- `shape`
- `info()`
- `describe()`
- `dtypes`
- `isnull().sum()`
- `duplicated()`
- `value_counts()`

Identified the dataset structure, data types, missing values and duplicate records.

### 2. Data Cleaning

- Checked for missing values across all columns.
- Identified one duplicate record.
- Removed duplicate rows using `drop_duplicates()`.
- Reset the DataFrame index.
- Verified the final dataset dimensions.

Final dataset: **1,337 rows × 7 columns**

### 3. Exploratory Data Analysis (EDA)

Performed univariate and bivariate analysis to understand data distributions and relationships.

**Visualizations included:**

- Histograms with KDE.
- Boxplots.
- Countplots.
- Scatterplots.
- Correlation heatmaps.
- Group-wise average charge comparisons.

Analyzed categorical variables such as sex, smoker, region and children against insurance charges.

### 4. Outlier Detection

Applied the Interquartile Range (IQR) method to identify outliers in medical insurance charges.

**Findings:**
- Total outliers identified: 139.
- Smokers among outliers: 136.
- Non-smokers among outliers: 3.

This analysis highlighted a strong association between smoking status and unusually high insurance charges.

Rather than automatically removing these observations, I investigated their relationship with smoking status.

### 5. Statistical Analysis & Hypothesis Testing

Applied statistical tests to understand whether different variables have significant relationships with insurance charges.

**A. Independent T-Test**

Compared insurance charges between male and female individuals.

- P-value: 0.0338
- Result: Statistically significant difference at the 5% significance level.

Also compared smokers and non-smokers.

- P-value: Approximately 1.41 × 10⁻²⁸².
- Result: Strong statistically significant difference in insurance charges.

**B. ANOVA Test**

Examined differences in average insurance charges across geographical regions.

- P-value: 0.032763.
- Result: Statistically significant differences exist between at least some regional group means.

Also performed ANOVA to investigate the relationship between the number of children and insurance charges.

- P-value: 0.006133.
- Result: Significant differences exist among at least some children groups.

**C. Tukey HSD Test**

Applied Tukey's Honestly Significant Difference test for pairwise regional comparisons to identify which regional groups differed significantly.

**D. Pearson Correlation**

Analyzed relationships between continuous variables and insurance charges.

| Feature | Correlation |
|---|---:|
| Age | 0.298 |
| BMI | 0.198 |

Both features showed positive correlations with insurance charges.

**E. Chi-Square Test**

Examined the association between sex and smoking status.

- Chi-square statistic: 7.469.
- P-value: 0.0063.
- Result: Smoking status and sex are statistically associated in this dataset.

### 6. Data Encoding & Feature Preparation

Converted categorical variables into numerical representations.

- Sex: Female = 0, Male = 1.
- Smoker: No = 0, Yes = 1.
- Region: Applied One-Hot Encoding.
- Used `drop_first=True` to avoid redundant dummy variables.

Created a correlation heatmap to examine relationships between the encoded features.

### 7. Initial Machine Learning Model

Implemented Multiple Linear Regression using Scikit-learn.

**Target Variable:** Charges

**Independent Variables:**
- Age
- Sex
- BMI
- Children
- Smoker
- Region (encoded variables)

**Train-Test Split:**
- Training: 80%
- Testing: 20%
- Random State: 0

**Initial Model Performance:**

| Metric | Result |
|---|---:|
| Training R² | 0.7489 |
| Testing R² | 0.7530 |
| Test MAE | 4,396.03 |
| Test RMSE | 6,445.64 |

### 8. OLS Regression & Feature Selection

Used Ordinary Least Squares (OLS) regression through Statsmodels to examine the statistical significance of individual predictors.

Applied Backward Elimination with a significance level of 0.05.

Features with higher p-values were progressively removed.

**Selected Features:**
- Age
- BMI
- Children
- Smoker

The reduced model maintained approximately the same predictive performance as the original model.

This demonstrated that a smaller feature set could achieve comparable results.

### 9. Multicollinearity Check using VIF

Calculated Variance Inflation Factor (VIF) to examine multicollinearity among independent variables.

The VIF values for the original encoded predictors were relatively close to 1.

**Finding:** No concerning multicollinearity was observed among the included predictors.

### 10. Model Comparison

Compared different feature combinations to understand their impact on predictive performance.

Models evaluated included:

- All 8 features.
- Features selected after backward elimination.
- Age, BMI and Smoker.
- Smoker only.
- Age and BMI without Smoker.

The analysis showed that smoking status was an especially important predictor of insurance charges.

### 11. Feature Engineering – Model Improvement

Introduced an additional interaction feature:

**BMI-Smoker Interaction**

Created a binary feature named `bmi30_smoker`.

Condition:

- BMI greater than or equal to 30.
- Individual is a smoker.

The feature captures the combined effect of higher BMI and smoking status.

This additional feature improved the predictive performance of the Linear Regression model.

**Improved Model Features:**
- Age
- BMI
- Children
- Smoker
- BMI-Smoker Interaction

### 12. Final Model Performance

| Metric | Initial Model | Improved Model |
|---|---:|---:|
| Training R² | 0.7489 | 0.8632 |
| Testing R² | 0.7530 | 0.8536 |

<ins>Test R² improved from 75.30% to 85.36%.</ins>

The engineered interaction feature helped the model explain a greater proportion of variation in insurance charges.

### 13. Residual Analysis

Performed residual analysis to understand prediction errors.

Created:
- Residuals vs Predicted scatterplot.
- Residual distribution histogram.

These visualizations help assess the model's error patterns and regression assumptions.

### 14. Custom Insurance Cost Prediction Function

Developed a custom Python function named `predict_charges()` using the trained model.

The function accepts:

- Age
- BMI
- Number of children
- Smoking status

It automatically:
1. Converts smoking status into a numerical value.
2. Generates the BMI-Smoker interaction feature.
3. Creates a DataFrame with the required model features.
4. Predicts the estimated annual medical insurance charges.

**Example Input:**

| Parameter | Value |
|---|---:|
| Age | 40 |
| BMI | 33 |
| Children | 2 |
| Smoker | Yes |

**Predicted Annual Insurance Charges: 42,048.42**

## Key Findings

- Smoking status is one of the strongest predictors of medical insurance charges.
- Most identified high-charge outliers were smokers.
- Age and BMI showed positive correlations with insurance charges.
- Statistical tests identified significant differences across certain categorical groups.
- Backward Elimination helped identify a smaller set of relevant predictors.
- VIF analysis showed no concerning multicollinearity.
- Feature engineering using the BMI-Smoker interaction improved test R² from 75.30% to 85.36%.

## Key Learnings

Through this project, I gained practical experience in:

- Data cleaning and preprocessing.
- EDA and statistical visualization.
- IQR-based outlier detection.
- Independent T-tests and ANOVA.
- Tukey HSD and Chi-square testing.
- Pearson correlation analysis.
- Categorical encoding.
- OLS regression.
- Multicollinearity analysis using VIF.
- Backward Elimination.
- Multiple Linear Regression.
- Feature engineering and model optimization.
- Model evaluation using R², MAE and RMSE.
- Residual analysis.
- Building custom prediction functions.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn jupyter
```

### 3. Open Jupyter Notebook

```bash
jupyter notebook
```

### 4. Load the Dataset

```python
import pandas as pd

data = pd.read_csv("insurance.csv")
```

### 5. Execute the Notebook

Run the notebook cells sequentially to reproduce the data cleaning, statistical analysis, feature selection, model training and prediction workflow.


*This project represents my progression from exploratory data analysis to statistical modeling and machine learning optimization. It demonstrates how statistical insights and thoughtful feature engineering can improve predictive performance.*
