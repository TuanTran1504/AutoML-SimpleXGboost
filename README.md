**AutoML-SimpleXGBoost**

A simplified AutoML pipeline that performs automated preprocessing and hyperparameter tuning using XGBoost for classification, regression, and multi-class problems. This project also includes data inspection tools, handling for time series, and GUI support (Tkinter or Flask-ready).

Features

✅ Automated handling of missing values (mean, median, zero, drop, mode, missing-label)

📊 Pre-analysis of missing value patterns in training and test datasets

📦 Support for binary, multi-class classification, and regression

📈 Log-transform option for regression targets

🔁 Time series split for sequential data

🔎 Auto-detection of problem type based on target distribution

🎯 Built-in hyperparameter optimization using Bayesian Search (BayesSearchCV)

🧪 Integration-ready with GUI (e.g., Tkinter, Flask)


Usage Instruction
Clone the repo
```bash
git clone https://github.com/TuanTran1504/AutoML-SimpleXGboost.git
cd AutoML-SimpleXGboost
```

🚀 Running the Flask Interface
```bash
# 1 Create Virtual Environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
```bash
# 2. Install dependencies
pip install -r requirements.txt
```
```bash
# 3. Run the Flask app
python app.py
```
Continue with open your train file (test file is currently not supported)
<img width="523" height="250" alt="image" src="https://github.com/user-attachments/assets/93591179-6157-4567-984d-e7a3f9021911" />

If there is missing data, you will need to choose the methods
<img width="499" height="229" alt="image" src="https://github.com/user-attachments/assets/1fd93464-ab37-47b7-be8c-95bc02665b2d" />

Else you will go to the train model page
<img width="541" height="365" alt="image" src="https://github.com/user-attachments/assets/795a864d-9c52-4108-8fe9-ac567fec0c0f" />



