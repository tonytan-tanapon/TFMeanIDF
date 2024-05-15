import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    test_data = pd.read_csv('data/test_data.csv')
    X_test = test_data['text']
    y_test = test_data['label']
    return X_test, y_test

def evaluate_model():
    X_test, y_test = load_data()
    
    pipeline = load('results/model.joblib')
    y_pred = pipeline.predict(X_test)
    
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    matrix = confusion_matrix(y_test, y_pred)
    
    with open('results/evaluation_results.txt', 'w') as f:
        f.write('Model Evaluation Results\n')
        f.write('='*24 + '\n')
        f.write(report + '\n')
        f.write('Confusion Matrix:\n')
        f.write(str(matrix) + '\n')
    
    print('Evaluation results saved to results/evaluation_results.txt')

if __name__ == '__main__':
    evaluate_model()
