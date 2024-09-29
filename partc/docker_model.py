import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix

def train_model(X, y, task='regression', model_type='linear', model_save_path='model.pkl'):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if task == 'regression':
        if model_type == 'linear':
            # Linear Regression
            model = LinearRegression()
        elif model_type == 'sgd':
            # SGD Regressor
            model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.000021)
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
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        elif model_type == 'log':
            model = LogisticRegression()
        elif model_type == 'random_forest':
            # Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100)
        elif model_type == 'decision_tree':
            # Decision Tree Classifier
            model = DecisionTreeClassifier()
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
