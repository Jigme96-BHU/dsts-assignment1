import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix

def train_model(X, y, task='', model_type=''):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if task == 'regression':
        if model_type == 'linear':
            # Linear Regression
            model = joblib.load('./model/linear_regression_model.pkl')
        elif model_type == 'sgd':
            # SGD Regressor
            model = joblib.load('./model/sgd_regression_model.pkl')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and return Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    
    elif task == 'classification':
        if model_type == 'mlp':
            # MLP Classifier
            model = joblib.load('./model/mlp_classification_model.pkl')

        elif model_type == 'random_forest':
            # Random Forest Classifier
            model = joblib.load('./model/random_forest_classification_model.pkl')
        elif model_type == 'decision_tree':
            # Decision Tree Classifier
            model = joblib.load('./model/decision_tree_classification_model.pkl')

        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and return accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test,y_pred)

        return accuracy,cm
    
    else:
        raise ValueError(f"Unknown task: {task}")
