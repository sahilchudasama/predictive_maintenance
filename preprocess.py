import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Convert datetime strings to datetime objects
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Extract useful features from datetime
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    
    # Drop the original timestamp column
    data = data.drop('timestamp', axis=1)
    
    # Preprocess data
    X = data.drop('failure', axis=1)  # Features
    y = data['failure']  # Target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test
