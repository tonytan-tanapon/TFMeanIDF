import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from model import TFMeanIDF

def load_data():
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data['text']
    y_train = train_data['label']
    return X_train, y_train

def train_model():
    X_train, y_train = load_data()
    
    pipeline = Pipeline([
        ('tfmeanidf', TFMeanIDF()),
        ('clf', LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    dump(pipeline, 'results/model.joblib')

if __name__ == '__main__':
    train_model()
