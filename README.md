# 📱 Customer Churn Prediction for Telecom Business

A comprehensive end-to-end machine learning project that predicts whether a telecom customer is likely to churn (cancel their subscription).

## 🎯 Project Overview

This project builds a predictive model to identify customers at risk of churning from a telecom company. By leveraging machine learning, we can:
- Identify at-risk customers early
- Implement targeted retention strategies
- Reduce revenue loss from customer churn
- Improve overall customer lifetime value

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- **Rows**: 7,043 customers
- **Columns**: 21 features
- **Target**: Binary classification (Churn: Yes/No)
- **Features**: Demographics, services used, charges, and contract details

## 🛠 Tech Stack

### Programming & Data Science
- **Python** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

### Deployment & Visualization
- **Streamlit** - Interactive web application
- **Streamlit Community Cloud** - Cloud deployment
- **GitHub** - Version control and repository

### Development Tools
- **JupyterLab** - Interactive notebook environment
- **Git & GitHub** - Version control

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── telecom_churn.csv              # Original dataset
│   └── cleaned_telco_churn.csv        # Cleaned dataset
│
├── notebooks/
│   └── EDA.ipynb                       # Exploratory Data Analysis & Model Building
│
├── models/
│   ├── churn_model.pkl                 # Trained XGBoost model
│   └── preprocessor.pkl                # Data preprocessor (scaling & encoding)
│
├── images/
│   └── feature_importance.png          # Feature importance visualization
│
├── app.py                              # Streamlit web application
├── feature_importance.csv              # Feature importance scores
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📈 Features

### 🔮 Prediction Tab
- Interactive customer input form
- Real-time churn predictions
- Churn probability scores
- Retention probability
- Risk level classification
- Actionable recommendations

### 📊 Feature Importance Tab
- Top 15 features visualization
- Ranked by importance score
- Key business insights
- Feature descriptions

### ℹ️ Model Information Tab
- Model algorithm details
- Training dataset information
- Model performance metrics
- Data preprocessing steps
- Business problem statement

### 📈 Analytics Tab
- EDA findings and patterns
- Business recommendations
- Customer behavior insights
- Retention strategies

## 🎓 Project Workflow

### 1. **Data Loading & Exploration**
   - Loaded 7,043 customer records
   - Analyzed 21 features
   - Checked for missing values and duplicates

### 2. **Data Cleaning**
   - Removed customerID (non-predictive)
   - Converted TotalCharges to numeric
   - Replaced blank values with appropriate defaults
   - Handled missing values
   - Removed duplicates
   - Saved cleaned dataset

### 3. **Exploratory Data Analysis (EDA)**
   - Analyzed churn distribution
   - Created visualizations for:
     - Demographic factors vs churn
     - Contract type impact
     - Internet service influence
     - Monthly & total charges distribution
     - Correlation heatmap
     - Feature distributions

### 4. **Feature Engineering**
   - Split features and target variable
   - Encoded target (Churn: 0/1)
   - Applied One-Hot Encoding to categorical variables
   - Scaled numerical features with StandardScaler
   - Used ColumnTransformer for pipeline
   - Train-Test Split (80/20)

### 5. **Model Building**
   - **Models Compared:**
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
   - **Evaluation Metrics:**
     - Accuracy
     - Precision
     - Recall
     - F1 Score

### 6. **Model Evaluation**
   - Classification report
   - Confusion matrix
   - ROC curve & ROC-AUC
   - Precision-Recall curve
   - Cross-validation

### 7. **Hyperparameter Tuning**
   - GridSearchCV for exhaustive search
   - RandomizedSearchCV for random sampling
   - Optimized parameters:
     - n_estimators (number of trees)
     - max_depth (tree depth)
     - min_samples_leaf
     - min_samples_split
     - max_features
   - Selected best model

### 8. **Feature Importance Analysis**
   - Extracted top features
   - Generated importance scores
   - Created visualizations
   - **Top 5 Features:**
     1. Tenure (months with company)
     2. Total Charges (lifetime value)
     3. Monthly Charges (service cost)
     4. Contract Type (Month-to-month vs Long-term)
     5. Tech Support (Yes/No)

## 🔍 Key Business Insights

### 1. Contract Impact
- **Month-to-month contracts**: Highest churn rate
- **One/Two-year contracts**: Significantly lower churn
- **Action**: Encourage multi-year contracts with incentives

### 2. Tenure Effect
- New customers churn more frequently
- Loyal customers (24+ months) have lower churn
- **Action**: Invest in better onboarding for new customers

### 3. Internet Service Type
- Fiber optic customers show higher churn
- Potential service quality issues
- **Action**: Investigate and improve fiber optic service quality

### 4. Charges & Affordability
- High monthly charges correlate with higher churn
- Price-sensitive customers need better value
- **Action**: Develop better value packages and bundled services

### 5. Support Services
- Tech support subscribers churn less
- Online security subscribers are more loyal
- **Action**: Bundle support services as default or heavily promote

## 📊 Model Performance

The final XGBoost model demonstrates:
- **High Accuracy** - Correctly classifies most churn cases
- **Balanced Precision & Recall** - Effective at identifying at-risk customers
- **Strong Generalization** - Good performance on unseen data
- **Feature Interpretability** - Clear understanding of churn drivers

## 🎯 Business Recommendations

### For Marketing & Sales
1. **Target retention campaigns** at identified high-risk customers
2. **Offer loyalty discounts** for multi-year contracts
3. **Bundle valuable services** (tech support, security) at competitive prices
4. **Create win-back campaigns** for churning customers

### For Operations & Support
1. **Improve service quality** for fiber optic customers
2. **Enhance onboarding** in the first year
3. **Provide 24/7 support** for critical customer segments
4. **Implement proactive monitoring** of customer satisfaction

### For Product Development
1. **Develop competitive pricing tiers** for budget-conscious customers
2. **Improve internet speed** for high-paying customers
3. **Create bundled service packages** to increase perceived value
4. **Add loyalty rewards program** for long-term customers

## 📱 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository and branch
5. Set main file as `app.py`
6. Click "Deploy"

## 📝 Usage Examples

### Example 1: New Customer
- Tenure: 1 month
- Monthly Charges: $70
- Contract: Month-to-month
- Tech Support: No
- **Prediction**: High churn risk ⚠️

### Example 2: Loyal Customer
- Tenure: 60 months
- Monthly Charges: $65
- Contract: Two-year
- Tech Support: Yes
- **Prediction**: Low churn risk ✅

## 🔐 Privacy & Ethics

- **Data Privacy**: Model trained on anonymized customer data
- **Fairness**: No discriminatory variables included
- **Transparency**: Feature importance clearly documented
- **Purpose**: Improve customer service, not enable discriminatory practices

## 📚 Files Description

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `EDA.ipynb` | Exploratory Data Analysis & model building |
| `churn_model.pkl` | Trained XGBoost model |
| `preprocessor.pkl` | Data preprocessing pipeline |
| `feature_importance.csv` | Feature importance scores |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- **Dataset**: IBM Telco Customer Churn Dataset
- **Libraries**: Pandas, Scikit-learn, XGBoost, Streamlit
- **Inspiration**: Real-world business problem solving with data science

---

**Project Status**: ✅ Complete & Production Ready

**Version**: 1.0.0
