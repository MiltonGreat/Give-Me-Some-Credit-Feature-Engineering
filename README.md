# Give Me Some Credit Feature Engineering

### Overview

This project focuses on building a predictive model for credit risk assessment using the "Give Me Some Credit" dataset from Kaggle. The goal is to predict whether a borrower will experience serious delinquency (90+ days past due) within the next two years.

The key innovation in this project is advanced feature engineering, where we create new predictive features from raw financial data to improve model performance. The engineered features capture complex relationships in credit behavior, delinquency patterns, and debt burden, leading to a more accurate risk assessment.

### Dataset

Target Variable: SeriousDlqin2yrs (1 = default within 2 years, 0 = no default)

Feature	Description
- RevolvingUtilizationOfUnsecuredLines: Credit card/utilization ratio
age	Borrower’s age
- NumberOfTime30-59DaysPastDueNotWorse: Late payments (30–59 days)
- DebtRatio	Monthly: debt payments / Monthly income
- MonthlyIncome:	Borrower’s income
- NumberOfOpenCreditLinesAndLoans:	Open credit lines
- NumberOfTimes90DaysLate	Severe late payments: (90+ days)
- NumberRealEstateLoansOrLines:	Mortgages/real estate loans
- NumberOfTime60-89DaysPastDueNotWorse:	Late payments (60–89 days)
- NumberOfDependents:	Number of dependents
- Engineered Features:	TotalPastDue, IncomePerDependent, DebtToIncome

### Methodology

1. Data Cleaning
- Handle missing values (median imputation)
- Cap extreme values (winsorization at 1st & 99th percentiles)
- Fix invalid entries (e.g., age=0 → median age)

2. Feature Engineering

Delinquency Aggregates:
- TotalPastDue (Sum of all late payments)
- WeightedDelinquencyScore (Severity-weighted delinquency metric)
- HasAnyDelinquency (Binary indicator)

Debt & Income Ratios:
- DebtToIncome
- IncomePerDependent
- Credit Utilization Enhancements:
- RevolvingUtilizationSquared
- RevolvingUtilizationLog

Interaction Features:
- UtilizationByDelinquency (Credit usage × late payments)
- AgeDebtRatio (Age × Debt burden)

Risk Indices:
- FinancialStressIndex
- CreditComplexityIndex

3. Feature Selection
- Multicollinearity Check (Remove highly correlated features)
- Random Forest Importance Ranking (Keep top predictive features)

4. Model Training & Evaluation

Algorithms tested:
- Logistic Regression (Baseline)
- Random Forest
- XGBoost / LightGBM

Evaluation Metrics:
- AUC-ROC (Primary metric for imbalanced data)
- Precision-Recall Curve
- Confusion Matrix

### Key Challenges

- Class imbalance (Few borrowers default, making prediction difficult)
- Missing values (e.g., MonthlyIncome, NumberOfDependents)
- Extreme outliers (e.g., unrealistic ages, extreme debt ratios)

### Key Findings

**Top 15 Most Important Features**:
Feature  Importance
- WeightedDelinquencyScore    0.148912
- TotalPastDue    0.134966
- UtilizationByDelinquency    0.130238
- HasAnyDelinquency    0.115453
- RevolvingUtilizationOfUnsecuredLines    0.073028
- RevolvingUtilizationOfUnsecuredLines_Scaled    0.061826
- RevolvingUtilizationSquared    0.051770
- RevolvingUtilizationLog    0.050622
- UtilizationToIncome    0.036853
- NumberOfTimes90DaysLate    0.030859
- HasPastDue90    0.027272
- NumberOfTime30-59DaysPastDueNotWorse    0.026253
- NumberOfTime60-89DaysPastDueNotWorse    0.019084
- FinancialStressIndex    0.016276
- DelinquencySeverityRatio    0.015096

**Insight**:
- Delinquency history is the strongest predictor.
- Engineered features dominate the top predictors, proving their value.
- Credit utilization (non-linear transformations) also plays a key role.

### Business Applications

- Banks & Lenders – Improve loan approval decisions by identifying high-risk borrowers.
- Credit Scoring Systems – Enhance traditional credit scores with dynamic behavioral features.
- Risk Management Teams – Monitor early warning signs (e.g., rising FinancialStressIndex).

### Conclusion

This project successfully demonstrated the power of strategic feature engineering in building a predictive credit risk model. By creating sophisticated features that capture nuanced patterns in borrower behavior, we significantly enhanced the model's ability to assess the likelihood of serious delinquency.

### Source

![Give Me Some Credit Dataset from Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
