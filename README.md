# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project uses machine learning to classify credit card transactions as normal or fraudulent. The work includes data exploration, preprocessing, training and comparing four classification models, evaluating their performance, and selecting a decision threshold for the Random Forest model.

## Dataset

- **Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Original dataset:** 284,807 transactions and 31 columns
- **Input features:** 30 columns (`Time`, `V1`–`V28`, and `Amount`)
- **Target:** `Class` (`0` = normal, `1` = fraud)

The dataset is highly imbalanced. After duplicate rows were removed, the notebook contains **283,726 transactions**:
- Normal transactions: 283,253 (about 99.83%)
- Fraudulent transactions: 473 (about 0.17%)

The dataset CSV is not included in this repository. Download it from Kaggle and place `creditcard.csv` in the project root before running the notebook.

## Work Performed

### Exploratory Data Analysis
- Inspected the dataset structure and descriptive statistics.
- Checked missing values, duplicate rows, and unique values.
- Examined the class distribution.
- Explored transaction amount and transaction time.
- Compared amount and time distributions between normal and fraudulent transactions.
- Examined feature correlations using a correlation heatmap and a chart of the ten most correlated features.

### Data Preprocessing
- Removed duplicate records.
- Separated the input features (`X`) and target (`y`).
- Split the data into training and testing sets using an 80:20 split and stratification.
- Standardized the `Time` and `Amount` columns. The scaler was fitted on the training data and applied to the test data.
- Used `class_weight='balanced'` in the classifiers that support it to account for class imbalance.

### Models Compared
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. HistGradientBoosting

### Evaluation
The models were compared using accuracy, precision, recall, and F1-score. ROC curves and Precision-Recall curves were also examined. The notebook calculates the reported PR-AUC summary using scikit-learn's `average_precision_score` (Average Precision).

## Model Comparison

The following scores are from the notebook's model-comparison results, using the models' default prediction thresholds.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.52% | 5.62% | 87.37% | 10.57% |
| Decision Tree | 99.53% | 23.20% | 77.89% | 35.75% |
| Random Forest | 99.95% | 94.52% | 72.63% | 82.14% |
| HistGradientBoosting | 99.82% | 47.83% | 81.05% | 60.16% |

### ROC-AUC and Precision-Recall Results

| Model | ROC-AUC | Average Precision (PR summary) |
|---|---:|---:|
| Logistic Regression | 0.9658 | 0.6719 |
| Decision Tree | 0.8886 | 0.5035 |
| Random Forest | 0.9391 | 0.8012 |
| HistGradientBoosting | 0.9577 | 0.7190 |

Random Forest achieved the highest F1-score and Average Precision among the four models in this comparison. Logistic Regression achieved the highest ROC-AUC.

## Random Forest Threshold Analysis

A separate validation portion of the training data was used to compare decision thresholds from 0.1 to 0.5. A threshold of **0.4** was selected based on the validation F1-score.

On the held-out test set, the Random Forest model at threshold 0.4 achieved:

- Accuracy: **99.95%**
- Precision: **93.33%**
- Recall: **73.68%**
- F1-score: **82.35%**

The test confusion matrix reported by the notebook is:

| | Predicted normal | Predicted fraud |
|---|---:|---:|
| Actual normal | 56,646 | 5 |
| Actual fraud | 25 | 70 |

Thus, the model identified 70 of the 95 fraudulent transactions in the test set. Accuracy alone should not be used to judge this project because fraudulent transactions are rare; precision and recall are also important.

## Project Files

- `Credit-card-fraud-detection.ipynb` – notebook containing the analysis, preprocessing, model training, evaluation, and model saving/loading steps.
- `Credit_Card_Fraud_Detection_Project_Report.pdf` – project report.
- `Credit_Card_Fraud_Detection_Presentation.pptx` – project presentation.
- `Models/random_forest_model.pkl` – saved Random Forest model.
- `Models/standard_scaler.pkl` – saved scaler for `Time` and `Amount`.
- `Results/model_comparison.csv` – model comparison results.
- `requirements.txt` – Python package requirements.
- `.gitignore` – excludes the dataset and temporary files from Git.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## How to Run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download `creditcard.csv` from the Kaggle dataset linked above.
4. Place `creditcard.csv` in the same folder as the notebook.
5. Open Jupyter Notebook and run `Credit-card-fraud-detection.ipynb`.

## Limitations and Further Work

This is an educational machine learning project using a historical, highly imbalanced dataset. Its results do not establish that the model is ready for real-world deployment. Further work could include testing on newer data, monitoring changes in transaction patterns, and evaluating threshold choices against the costs of missed fraud and false alerts.

## Author

**Surajit Bera**  
Data Science Internship Project – July Batch
