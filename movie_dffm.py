# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names
import tensorflow as tf

def load_data():
    # Load MovieLens dataset or any other dataset
    # Example: df = pd.read_csv('movielens_dataset.csv')
    # Preprocess the data as needed
    # Example: df = preprocess_data(df)
    return df

def preprocess_data(df):
    # Perform data preprocessing steps such as handling missing values,
    # encoding categorical variables, and feature engineering
    # Example: df = handle_missing_values(df)
    # Example: df = encode_categorical_variables(df)
    return df

def train_model(df):
    # Define feature columns
    sparse_features = ['user_id', 'movie_id', 'genre']
    feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                       for feat in sparse_features]

    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Create input data for the model
    train_model_input = {name: train[name].values for name in sparse_features}
    test_model_input = {name: test[name].values for name in sparse_features}

    # Define the DFFM model
    model = DeepFM(feature_columns, task='regression')

    # Compile the model
    model.compile("adam", "mse", metrics=['mse'], )

    # Train the model
    model.fit(train_model_input, train['rating'].values, batch_size=256, epochs=10,
              validation_split=0.2, )

    # Evaluate the model
    result = model.evaluate(test_model_input, test['rating'].values, verbose=2)
    print("Test MSE: {:.4f}".format(result))

    # Save the model
    model.save('dffm_model.h5')

if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    # Train the DFFM model
    train_model(df)
