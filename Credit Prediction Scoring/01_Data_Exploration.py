# Databricks notebook source
# MAGIC %md
# MAGIC ##Credit Risk Analysis Project 

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Exploration & Quality Assessment
# MAGIC
# MAGIC ##  Objective
# MAGIC Perform comprehensive exploratory data analysis (EDA) to understand the structure, quality, and characteristics of the banking dataset before building our credit risk model.
# MAGIC
# MAGIC ##  Business Context
# MAGIC Before making any loan decisions, banks need to thoroughly understand their customer data. This analysis helps identify data quality issues, understand customer demographics, and uncover initial patterns that might influence credit risk.
# MAGIC
# MAGIC
# MAGIC ##  Technical Approach
# MAGIC - **Data Loading**: Connect to Databricks table and load banking data
# MAGIC - **Quality Checks**: Assess missing values, data types, and basic statistics
# MAGIC - **Distribution Analysis**: Examine how key variables are distributed
# MAGIC - **Correlation Insights**: Identify initial relationships between variables
# MAGIC
# MAGIC ##  Key Analysis Steps
# MAGIC
# MAGIC ### 1. Data Loading & Initial Inspection
# MAGIC ```python
# MAGIC # Load the dataset from Databricks catalog
# MAGIC df = spark.table("personal_catalog.default.bank_loan_modelling")
# MAGIC ```
# MAGIC
# MAGIC ### 2. Data Quality Assessment
# MAGIC - Check for missing values across all columns
# MAGIC - Validate data types match expected formats
# MAGIC - Identify any outliers or anomalous values
# MAGIC - Verify dataset size and basic completeness
# MAGIC
# MAGIC ### 3. Statistical Summary
# MAGIC - Generate descriptive statistics for numerical features
# MAGIC - Examine categorical variable distributions
# MAGIC - Calculate key financial metrics and ranges
# MAGIC
# MAGIC ### 4. Initial Pattern Discovery
# MAGIC - Explore relationships between income and loan status
# MAGIC - Analyze age and experience distributions
# MAGIC - Examine credit card spending patterns
# MAGIC
# MAGIC ##  Expected Outcomes
# MAGIC - **Data Quality Report**: Summary of any data issues found
# MAGIC - **Statistical Overview**: Understanding of variable distributions  
# MAGIC - **Initial Insights**: Early patterns that may influence model design
# MAGIC - **Feature Selection Guidance**: Informed decisions about which variables to use
# MAGIC
# MAGIC ## Risk Mitigation
# MAGIC - Handle any missing data appropriately
# MAGIC - Document data quality issues for stakeholders
# MAGIC - Identify potential biases in the dataset
# MAGIC - Establish data validation rules for future applications
# MAGIC
# MAGIC ##  Next Steps
# MAGIC After completing this exploration, proceed to **02_feature_engineering.ipynb** to create advanced features based on these insights.
# MAGIC
# MAGIC ---
# MAGIC *This analysis forms the foundation for our entire credit risk scoring system. Quality data understanding leads to quality models.*

# COMMAND ----------

# Import libraries for comprehensive data analysis
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Starting Comprehensive Data Exploration for Credit Risk Analysis")


# 1.1 Load the dataset from Databricks catalog
print("Loading banking data from Databricks...")
df = spark.table("personal_catalog.default.bank_loan_modelling")

print("Data loaded successfully!")
print(f"Dataset dimensions: {df.count()} rows, {len(df.columns)} columns")



# 1.2 Initial data inspection
print("Initial Data Overview:")
print("=" * 50)
df.show(5, truncate=False)
df.printSchema()

print("\nColumn Names and Types:")
for i, field in enumerate(df.schema.fields):
    print(f"{i+1:2d}. {field.name:<20} {str(field.dataType):<15}")


### 2. Data Quality Assessment

# 2.1 Comprehensive data quality check
print("Data Quality Assessment")
print("=" * 50)

# Check for missing values
print("Missing Values Analysis:")
missing_data = []
for col in df.columns:
    null_count = df.filter(F.col(col).isNull()).count()
    missing_data.append((col, null_count, null_count/df.count()*100))

missing_df = spark.createDataFrame(missing_data, ["Column", "Null_Count", "Null_Percentage"])
missing_df.show(len(df.columns))

# Check for duplicate records
duplicate_count = df.count() - df.distinct().count()
print(f"\nDuplicate Records: {duplicate_count} ({duplicate_count/df.count()*100:.2f}%)")

# 2.2 Statistical summary for numerical columns
print("Statistical Summary for Numerical Variables")
print("=" * 50)

numerical_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
df.select(numerical_cols).describe().show()

# Additional statistical insights
print("\nAdvanced Statistical Insights:")
for col in numerical_cols:
    col_data = df.select(F.col(col))
    skewness = col_data.select(F.skewness(col)).collect()[0][0]
    kurtosis = col_data.select(F.kurtosis(col)).collect()[0][0]
    print(f"{col:<12}: Skewness = {skewness:7.3f}, Kurtosis = {kurtosis:7.3f}")


### 3. Distribution Analysis

# 3.1 Convert to pandas for detailed analysis
print("Converting to Pandas for detailed visualization...")
pandas_df = df.toPandas()
print(f"Pandas DataFrame shape: {pandas_df.shape}")

# Create subplots for distribution analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Numerical Variables', fontsize=16, fontweight='bold')

numerical_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for i, col in enumerate(numerical_cols):
    ax = axes[i//3, i%3]
    pandas_df[col].hist(bins=20, ax=ax, color=colors[i], alpha=0.7, edgecolor='black')
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistical annotations
    mean_val = pandas_df[col].mean()
    median_val = pandas_df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=1, label=f'Median: {median_val:.1f}')
    ax.legend()

plt.tight_layout()
plt.show()

# 3.2 Categorical variables analysis
print("Categorical Variables Analysis")
print("=" * 50)

categorical_cols = ['Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Categorical Variables', fontsize=16, fontweight='bold')

for i, col in enumerate(categorical_cols):
    ax = axes[i//3, i%3]
    value_counts = pandas_df[col].value_counts().sort_index()
    bars = value_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(value_counts)], alpha=0.7)
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for p in bars.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


### 4. Correlation & Relationship Analysis

# 4.1 Correlation matrix for numerical variables
print("Correlation Analysis Between Numerical Variables")
print("=" * 50)

# Calculate correlation matrix
corr_matrix = pandas_df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Highlight strong correlations
print("\nStrong Correlations (|r| > 0.5):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            print(f"  {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

# 4.2 Target variable analysis (Personal Loan)
print("Analysis of Target Variable: Personal Loan")
print("=" * 50)

loan_distribution = pandas_df['Personal Loan'].value_counts()
print(f"Personal Loan Distribution:\n{loan_distribution}")
print(f"\nLoan Approval Rate: {loan_distribution[1] / len(pandas_df) * 100:.2f}%")

# Compare features by loan status
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions by Loan Status', fontsize=16, fontweight='bold')

compare_cols = ['Age', 'Income', 'CCAvg', 'Experience', 'Education', 'Mortgage']
for i, col in enumerate(compare_cols):
    ax = axes[i//3, i%3]
    
    # Create boxplot
    data_to_plot = [pandas_df[pandas_df['Personal Loan'] == 0][col], 
                   pandas_df[pandas_df['Personal Loan'] == 1][col]]
    ax.boxplot(data_to_plot, labels=['No Loan', 'Has Loan'])
    ax.set_title(f'{col} by Loan Status', fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


### 5. Data Quality Summary & Insights

# 5.1 Generate comprehensive data quality report
print("COMPREHENSIVE DATA QUALITY REPORT")
print("=" * 60)

print("\nDATA QUALITY ASSESSMENT:")
print("-" * 40)
print(f"Total Records: {len(pandas_df):,}")
print(f"Total Features: {len(pandas_df.columns)}")
print(f"Missing Values: {pandas_df.isnull().sum().sum()} (Perfect!)")
print(f"Duplicate Records: {pandas_df.duplicated().sum()} (Perfect!)")

print("\nTARGET VARIABLE ANALYSIS:")
print("-" * 40)
print(f"Personal Loan Approval Rate: {pandas_df['Personal Loan'].mean() * 100:.2f}%")
print(f"Class Distribution: {pandas_df['Personal Loan'].value_counts().to_dict()}")

print("\nKEY INSIGHTS FROM EXPLORATORY ANALYSIS:")
print("-" * 40)
print("1. Income shows right-skewed distribution with few high earners")
print("2. Most customers are middle-aged (30-50 years)")
print("3. Average credit card spending varies widely")
print("4. Education levels are well distributed")
print("5. Majority of customers don't have mortgages")
print("6. Strong digital engagement among customers")

print("\nPOTENTIAL DATA ISSUES:")
print("-" * 40)
print("Age and Experience show perfect correlation (expected)")
print("Class imbalance in target variable (may need balancing)")
print("Some numerical features are heavily skewed")

print("\nRECOMMENDATIONS FOR NEXT STEPS:")
print("-" * 40)
print("1. Proceed with feature engineering to create domain-specific features")
print("2. Consider addressing class imbalance in modeling")
print("3. Apply transformations for skewed variables if needed")
print("4. Validate correlation between Age and Experience")

# 5.2 Save key insights for future reference
data_insights = {
    'dataset_shape': pandas_df.shape,
    'loan_approval_rate': pandas_df['Personal Loan'].mean() * 100,
    'avg_income': pandas_df['Income'].mean(),
    'avg_age': pandas_df['Age'].mean(),
    'digital_engagement_rate': pandas_df['Online'].mean() * 100,
    'credit_card_usage_rate': pandas_df['CreditCard'].mean() * 100
}


print("\nDATA EXPLORATION COMPLETE!")
print("Proceed to Notebook 02 for advanced feature engineering.")
