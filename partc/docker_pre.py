# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_data(file_path, task='regression'):

    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop irrelevant columns
    data = data.drop(['address','lat','lng','phone','link','title','color','type','cost_2','cuisine_color','groupon'], axis=1)
    
    
    # Drop duplicates
    final_data = data.drop_duplicates(inplace=False)  
      
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

   
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output=False to return a dense array

    # Fit and transform the data (assuming 'final_data' is a DataFrame)
    data_encoded = encoder.fit_transform(final_data[['cuisine', 'subzone']])

    # Convert to DataFrame for readability (optional)
    encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(['cuisine', 'subzone']))

    final_data_encoded = pd.concat([encoded_df, final_data], axis=1)

    final_data_encoded = final_data_encoded.drop(['cuisine','subzone'],axis=1)
    final_data_encoded = final_data_encoded.dropna()

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
        pca = PCA(n_components=95)  # Keep components explaining 95% variance
        X_scaled = pca.fit_transform(X_scaled)

    return X_scaled, y
