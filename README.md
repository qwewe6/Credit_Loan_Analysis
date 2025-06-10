# Credit Loan Type Prediction with Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Tools and Technologies Used](#tools-and-technologies-used)
- [Data Inspection and Cleanup](#data-inspection-and-cleanup)
- [Data Analysis](#data-analysis)
- [Machine Learning Models](#machine-learning-models)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
- [Model Evaluation](#model-evaluation)
- [Prediction for New Customer](#prediction-for-new-customer)
- [Confusion Matrices](#confusion-matrices)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [Resources](#resources)

---

## Project Overview
Using a dataset of 1,000 records sourced from Kaggle, our team applied machine learning techniques to predict loan types for individuals based on various demographic and credit-related factors. The primary objective was to investigate patterns in the data, clean and preprocess it, and evaluate multiple machine learning models for loan classification.

---

## Dataset Description
- **Source**: [Kaggle Credit Scoring Data](https://www.kaggle.com/datasets/cs49adityarajsharma/credit-scoring-data?select=credit_scoring.csv)
- **Records**: 1,000
- **Features**:
  - Demographics: Age, Gender, Marital Status, Education Level, Employment Status
  - Credit Factors: Credit Utilization Ratio, Payment History, Number of Credit Accounts, Loan Amount, Interest Rate, Loan Term, Type of Loan

---

## Tools and Technologies Used
- Python (Jupyter Notebook)
- pandas, numpy, matplotlib, seaborn
- scikit-learn (StandardScaler, LabelEncoder, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV)
- PowerPoint (Presentation)
- chatGPT (Code assistance, documentation)

---

## Data Inspection and Cleanup
- Checked data types and missing values
- Converted numeric columns like Payment History and Loan Amount to `int64`
- Encoded categorical variables
- Created a new column "Age Group" for bucketing ages (twenties, thirties, etc.)

---

## Data Analysis
- Analyzed average loan amount by education level

  ![Avg Loan Amount](/Resources/Avg%20Loan%20Amount%20Per%20Education%20Status.png)

- Compared average term lengths across loan types

  ![Avg Term Length](/Resources/Avg%20Term%20Length%20Per%20Type%20Of%20Loan.png)

- Compared loan counts by gender

  ![Loan Count By Gender](/Resources/Loan%20Count%20By%20Gender.png)

---

## Machine Learning Models

### Decision Tree Classifier
- Features and target defined
- Categorical features encoded
- Data split into training and test sets
- Trained and predicted using a Decision Tree model
- Used GridSearchCV to optimize hyperparameters

#### Decision Tree Evaluation
- Initial model performance:

  ![Decision Tree Report](/Resources/Decision%20Tree%20Classification%20Report.png)

- Improved model performance with hyperparameter tuning:

  ![Decision Tree Upgraded](/Resources/Decision%20Tree%20Upgraded%20Classification%20Report.png)

- Feature importance from the Decision Tree:

  ![Decision Tree Feature Importance](/Resources/Decision%20Tree%20Feature%20Importance.png)

### Random Forest Classifier
- Similar training steps as Decision Tree
- Hyperparameter tuning with GridSearchCV
- Predictions made on test data

#### Random Forest Evaluation
- Initial classification report:

  ![Random Forest Report](/Resources/Random%20Forest%20Classification%20Report.png)

- Improved classification report after tuning:

  ![Random Forest Upgraded](/Resources/Random%20Forest%20Upgraded%20Classification%20Report.png)

- Feature importance from the Random Forest:

![Random Forest Feature Importance](/Resources/Random%20Forest%20Feature%20Importance.png)

---

## Model Evaluation
- Accuracy remained under 40% even after optimization
- Precision, recall, and f1-score analyzed for all classes

---

## Prediction for New Customer
New customer data used to test both models:

```python
new_customer = {
    'Age': 34,
    'Gender': 'Female',
    'Marital Status': 'Single',
    'Education Level': 'Bachelor',
    'Employment Status': 'Employed',
    'Credit Utilization Ratio': 0.45,
    'Payment History': 2500,
    'Number of Credit Accounts': 3,
    'Age Group': 'thirties'
}
```

- Decision Tree Prediction: **Auto Loan**
- Random Forest Prediction: **Personal Loan**

![Loan Prediction](/Resources/Loan%20Prediction.png)

---

## Confusion Matrices
- Confusion matrices for both models:

  ![Confusion Matrices](/Resources/Confusion%20Matrices.png)

---

## Conclusion
- Both models had low accuracy (< 40%)
- The dataset appears to be randomly generated and lacks strong correlation between features and loan type
- Highlights importance of quality data and feature selection
- Future improvements:
  - Use real-world datasets
  - Try more complex models like Gradient Boosting or Neural Networks
  - Engineer more insightful features

---

## Contributors
- Max Becker
- Michael Bowman
- Lance Cannon
- Adrian Williams

---

## Resources
- Dataset from [Kaggle](https://www.kaggle.com/datasets/cs49adityarajsharma/credit-scoring-data?select=credit_scoring.csv)
- [chatGPT](https://chat.openai.com) for ideation, code assistance, and documentation
- Python, scikit-learn, Jupyter Notebook, PowerPoint