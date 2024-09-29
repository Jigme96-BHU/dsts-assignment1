import os
from preprocessing import preprocess_data
from modeling import train_model

# Path to the dataset
file_path = '../data/zomato_df_final_data.csv'

# Create the 'model' directory if it doesn't exist
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Choose task (either 'regression' or 'classification')
input_task = input('Choose number (1 or 2): 1 - regression or 2 - classification: ').strip()

# Convert the input to the appropriate task
if input_task == '1':
    task = 'regression'
elif input_task == '2':
    task = 'classification'
else:
    print("Invalid choice. Please enter 1 for regression or 2 for classification.")
    exit()

# Preprocess the data
X, y = preprocess_data(file_path, task=task)

# Choose model type based on task
if task == 'regression':
    model_type = 'linear'  # Choose between 'linear' and 'sgd'
    model_save_path = os.path.join(model_dir, f'{model_type}_regression_model.pkl')
    mse = train_model(X, y, task=task, model_type=model_type, model_save_path=model_save_path)
    print(f"Mean Squared Error for {model_type.capitalize()} Regression: {mse}")

    model_type = 'sgd'  # Choose between 'linear' and 'sgd'
    model_save_path = os.path.join(model_dir, f'{model_type}_regression_model.pkl')
    mse = train_model(X, y, task=task, model_type=model_type, model_save_path=model_save_path)
    print(f"Mean Squared Error for {model_type.capitalize()} Regression: {mse}")

elif task == 'classification':
    model_type = 'random_forest'  # Choose between 'mlp', 'random_forest', and 'decision_tree'
    model_save_path = os.path.join(model_dir, f'{model_type}_classification_model.pkl')
    accuracy, cm = train_model(X, y, task=task, model_type=model_type, model_save_path=model_save_path)
    print(f"Accuracy for {model_type.replace('_', ' ').capitalize()} Classification: {accuracy}")
    print(f"Confusion Matrix for {model_type.replace('_', ' ').capitalize()} Classification:")
    print(cm)

    model_type = 'mlp'  # Choose between 'mlp', 'random_forest', and 'decision_tree'
    model_save_path = os.path.join(model_dir, f'{model_type}_classification_model.pkl')
    accuracy, cm = train_model(X, y, task=task, model_type=model_type, model_save_path=model_save_path)
    print(f"Accuracy for {model_type.replace('_', ' ').capitalize()} Classification: {accuracy}")
    print(f"Confusion Matrix for {model_type.replace('_', ' ').capitalize()} Classification:")
    print(cm)

    model_type = 'decision_tree'  # Choose between 'mlp', 'random_forest', and 'decision_tree'
    model_save_path = os.path.join(model_dir, f'{model_type}_classification_model.pkl')
    accuracy, cm = train_model(X, y, task=task, model_type=model_type, model_save_path=model_save_path)
    print(f"Accuracy for {model_type.replace('_', ' ').capitalize()} Classification: {accuracy}")
    print(f"Confusion Matrix for {model_type.replace('_', ' ').capitalize()} Classification:")
    print(cm)
