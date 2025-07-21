# app.py
import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from loading import pre_loading, loading, model, regression_model, multiclass_model, detect_class_type, check_data_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
app = Flask(__name__)
app.secret_key = 'secret'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global_data = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stage = request.form.get('stage')

        # Stage 1: File upload
        if stage == 'upload' or ('train_file' in request.files and stage is None):
            train_file = request.files['train_file']
            test_file = request.files.get('test_file')

            train_path = os.path.join(UPLOAD_FOLDER, train_file.filename)
            train_file.save(train_path)
            test_path = None
            if test_file and test_file.filename:
                test_path = os.path.join(UPLOAD_FOLDER, test_file.filename)
                test_file.save(test_path)

            global_data['train_path'] = train_path
            global_data['test_path'] = test_path

            # Detect missing values
            train_df, test_df, train_na, test_na = pre_loading(train_path, test_path)
            columns = train_df.columns.tolist()
            global_data['columns'] = columns

            if train_na.empty:
                global_missing = "No Missing Data."
                return render_template(
                    'index.html', stage='target', columns=columns,
                    global_train_path=train_path, global_test_path=test_path or ''
                )
            else: 
                num_missing = train_na[train_na["Type"] == "Numerical"]
                cat_missing = train_na[train_na["Type"] == "Categorical"]
                if not num_missing.empty and not cat_missing.empty:
                    global_missing = "Missing Numerical and Categorical"
                elif not num_missing.empty:
                    global_missing = "Missing Numerical"
                elif not cat_missing.empty:
                    global_missing = "Missing Categorical"
                global_data['missing'] = global_missing
                return render_template(
                    'index.html', stage='imputation', columns=columns,
                    global_train_path=train_path, global_test_path=test_path or '', missing=global_missing
                )

                

        # Stage 2: Imputation chosen → move to target selection
        elif stage == 'imputation':
            num_method = request.form['num_method']
            cat_method = request.form['cat_method']
            train_path = request.form['train_path']
            test_path = request.form.get('test_path')

            global_data['num_method'] = num_method
            global_data['cat_method'] = cat_method

            columns = global_data['columns']
            data_check = check_data_loss(
                train_path, test_path,
                num_method, cat_method,
            )
            if data_check is False:
                error_msg = "Too much data loss after imputation. Please choose different methods."
                global_missing = global_data.get('missing', '')
                return render_template(
                    'index.html', stage='imputation', columns=columns,
                    global_train_path=train_path, global_test_path=test_path or '',
                    num_method=num_method, cat_method=cat_method, error=error_msg, missing=global_missing
                )
            else:
                return render_template(
                    'index.html', stage='target', columns=columns,
                    global_train_path=train_path, global_test_path=test_path or '',
                    num_method=num_method, cat_method=cat_method
                )

        # Stage 3: Target and model settings → train or error if mismatch
        elif stage == 'train':
            train_path = request.form['train_path']
            test_path = request.form.get('test_path')
            num_method = request.form['num_method']
            cat_method = request.form['cat_method']
            target = request.form['target_column']
            feature_columns = request.form.getlist('feature_columns')
            prob_type = request.form['problem_type']
            is_ts = request.form.get('is_time_series') == 'on'
            time_col = request.form.get('time_column') if is_ts else None
            n_iter = int(request.form.get('n_iter', 20))
            cv = int(request.form.get('cv', 3))
            # Preprocess data
            df = pd.read_csv(train_path)
            columns = df.columns.tolist()
            try:
                train_df = loading(
                    train_path,feature_columns=feature_columns,
                num_method=num_method, cat_method=cat_method,
                    target_column=target
                )
                if test_path:
                    val_df = loading(
                        test_path, feature_columns=feature_columns,
                        num_method=num_method, cat_method=cat_method,
                        target_column=target
                    )
                    X_val = val_df.drop(columns=target)
                    y_val = val_df[target]
            except ValueError as e:
                return render_template(
                    "index.html",
                    stage="target",
                    error=str(e),
                    columns=columns,
                    global_train_path=train_path,
                    global_test_path=request.form.get("test_path"),
                    num_method=request.form.get("num_method"),
                    cat_method=request.form.get("cat_method"),
                    target_column=target,
                    feature_columns=feature_columns
                )


            
            X = train_df.drop(columns=target)
            y = train_df[target]
                
            # Detect actual problem type
            detected_type = detect_class_type(y)
            if detected_type != prob_type:
                # Inform user of mismatch and return to target selection
                error_msg = f"Detected problem type '{detected_type}' does not match your selection '{prob_type}'. Please choose the correct type."
                columns = global_data['columns']
                return render_template(
                    'index.html', stage='target', columns=columns,
                    global_train_path=train_path, global_test_path=test_path or '',
                    num_method=num_method, cat_method=cat_method,
                    error=error_msg
                )

            # Initialize and train model
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if prob_type=="binary":
                print("Using binary classification model")
                unique_values = sorted(y.unique())
                #Chekc for binary classification
                if unique_values == [0, 1]:
                    y_encoded = y
                else:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    y_val_encoded = le.transform(y_val) if test_path else None
                # Tách tập train/validation
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=8)

                # Khởi tạo và train model
                opt = model(cat_cols, num_cols=num_cols, n_iter=n_iter, cv=cv)
                opt.fit(X_train, y_train)

                # Hiển thị kết quả
                print("Best cross_validation estimator:", opt.best_estimator_)
                print("Best training score:", opt.best_score_)
                
                y_pred = opt.predict(X_test)
                roc = roc_auc_score(y_test, y_pred)
                print("ROC_AUC:", roc)
                if test_path:
                    y_val_pred = opt.predict(X_val)
                    val_roc = roc_auc_score(y_val_encoded, y_val_pred)
                    print("Validation ROC_AUC:", val_roc)
                    return render_template('results.html',
                                           problem_type='binary',
                                           best_score=opt.best_score_,
                                           roc_auc=roc,
                                           val_roc_auc=val_roc)
                return render_template('results.html',
                                       problem_type='binary',
                                       best_score=opt.best_score_,
                                       roc_auc=roc)
            elif prob_type == "multiclass":

                cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

                #Model for multiclass classification
                opt, train_df=multiclass_model(cat_cols=cat_cols, target_column=target, train=train_df, num_cols=num_cols, n_iter=n_iter, cv=cv)
                print("Using multiclass model")
                X = train_df.drop(columns=target)
                y = train_df[target]
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify= y_encoded, random_state=8)
                opt.fit(X_train, y_train)
                print(opt.best_estimator_)
                print(opt.best_score_)
                metric_name = opt.scoring._score_func.__name__
                y_pred = opt.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy_score(y_test, y_pred))
                return render_template('results.html',
                                       problem_type='multiclass',
                                       best_score=opt.best_score_,
                                       accuracy=acc)
  
            else:
                print("Using regression model")

                if is_ts:
                    train_df['target_lag_1'] = train_df[target].shift(1)
                    train_df['target_lag_2'] = train_df[target].shift(2)
                    train_df.dropna(inplace=True)
                    train_df[time_col] = pd.to_datetime(train_df[time_col])
                    train_df['month'] = train_df[time_col].dt.month
                    train_df['weekday'] = train_df[time_col].dt.weekday
                    train_df['year'] = train_df[time_col].dt.year
                    train_df.sort_values(by=time_col, inplace=True)
                    train_df.drop(columns=[time_col], inplace=True)
                    split_index = int(len(train_df) * 0.8)
                    train = train_df.iloc[:split_index]
                    test = train_df.iloc[split_index:]

                    X_train = train.drop(columns=[target])
                    y_train = train[target]
                    X_test = test.drop(columns=[target])
                    y_test = test[target]
                else:
                    X = train_df.drop(columns=target)
                    y = train_df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                #Model for regression
                opt=regression_model(cat_cols=cat_cols, time_series=is_ts, num_cols=num_cols, log_transform=True, n_iter=n_iter, cv=cv)
                opt.fit(X_train, y_train)
                print(opt.best_estimator_)

                y_pred = opt.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                print("MAE:", mean_absolute_error(y_test, y_pred))
                print("MSE:", mean_squared_error(y_test, y_pred))
                return render_template('results.html',
                                       problem_type='regression',
                                       best_score=opt.best_score_,
                                       mae=mae, mse=mse)
        

    # Default GET: upload stage
    return render_template('index.html', stage='upload')

if __name__ == '__main__':
    app.run(debug=True)