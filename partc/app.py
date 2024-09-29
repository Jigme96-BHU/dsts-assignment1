import os
from docker_pre import preprocess_data
from docker_model import train_model

# Path to the dataset
file_path = './data/zomato_df_final_data.csv'

# Define task: Choose between 'regression' and 'classification'
task = 'classification'
# Preprocess the data
X, y = preprocess_data(file_path, task=task)

# Choose model type based on task
if task == 'regression':
    for model_type in ['linear', 'sgd']:
        mse = train_model(X, y, task=task, model_type=model_type)
        print(f"Mean Squared Error for {model_type.capitalize()} Regression: {mse}")


elif task == 'classification':
   for model_type in ['random_forest', 'mlp', 'decision_tree']:
        accuracy, cm = train_model(X, y, task=task, model_type=model_type)
        print(f"Accuracy for {model_type.replace('_', ' ').capitalize()} Classification: {accuracy}")
        print(f"Confusion Matrix for {model_type.replace('_', ' ').capitalize()} Classification:")
        print(cm)
