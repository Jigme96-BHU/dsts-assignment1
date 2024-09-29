import os
from docker_pre import preprocess_data
from docker_model import train_model


def Classification_model():
   # Path to the dataset
   file_path = './data/zomato_df_final_data.csv'

   # Define task: Choose between 'regression' and 'classification'
   task = 'classification'
   # Preprocess the data
   X, y = preprocess_data(file_path, task=task)

   if task == 'classification':
      for model_type in ['log','random_forest', 'mlp', 'decision_tree']:
         accuracy, cm = train_model(X, y, task=task, model_type=model_type)
         print(f"Accuracy for {model_type.replace('_', ' ').capitalize()} Classification: {accuracy}")
         print(f"Confusion Matrix for {model_type.replace('_', ' ').capitalize()} Classification:")
         print(cm)
