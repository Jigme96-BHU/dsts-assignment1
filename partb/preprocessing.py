# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(file_path, task='regression'):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop irrelevant columns
    data = data.drop(['address', 'link', 'title', 'color', 'type', 'phone', 'cuisine_color'], axis=1)
    
    # Drop more irrelevant columns
    data = data.drop(['lat', 'lng', 'groupon', 'cost_2'], axis=1)
    
    # Handle categorical variables
    categorical = [var for var in data.columns if data[var].dtype == 'O']
    
    # Find numerical variables
    numericals = [var for var in data.columns if data[var].dtype != 'O']
    
    # Separate numerical and categorical data
    num_data = data[numericals]
    cat_data = data[categorical]
    
    # Concatenate numerical and categorical data
    final_data = pd.concat([num_data, cat_data], axis=1)
    
    # Drop duplicates
    final_data.drop_duplicates(inplace=True)
    
    # Handle missing values
    final_data['cost'].fillna(final_data['cost'].median(), inplace=True)
    final_data.dropna(subset=['rating_number', 'votes'], inplace=True)
    
        # Step 2: Simplify the 'rating_text' column into two classes
    def simplify_ratings(rating):
        if rating in ['Poor', 'Average']:
            return 1  # Class 1
        elif rating in ['Good', 'Very Good', 'Excellent']:
            return 2  # Class 2
        else:
            return None  # Exclude other records

    final_data['binary_rating'] = final_data['rating_text'].apply(simplify_ratings)

    # Encoding categorical variables
    final_data_encoded = pd.get_dummies(final_data, columns=['subzone', 'cuisine'], prefix=['subzone', 'cuisine'])


    if task == 'classification':
        # Simplify classification into binary
        final_data_encoded.dropna(subset=['binary_rating'], inplace=True)
        X = final_data_encoded.drop(columns=['binary_rating','rating_number','rating_text'])  # Features
        y = final_data_encoded['binary_rating']  # Binary Target
    else:
        # For regression
        X = final_data_encoded.drop(columns=['rating_number','binary_rating','rating_text'])  # Features
        y = final_data_encoded['rating_number']  # Target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if regression is chosen
    if task == 'regression':
        pca = PCA(n_components=0.95)  # Keep components explaining 95% variance
        X_scaled = pca.fit_transform(X_scaled)

    return X_scaled, y
