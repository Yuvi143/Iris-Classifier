import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def train_model():
    print("Initializing artifact directory structures...")
    os.makedirs("artifacts", exist_ok=True)
    
    iris_local_path = 'data/Iris.csv'
    if not os.path.exists(iris_local_path):
        raise FileNotFoundError(
            f"Missing required dataset: {iris_local_path}. "
            "Please run your 'dvc import' command first."
        )
        
    print("Loading and preprocessing data...")

    # Load dataset
    df = pd.read_csv(iris_local_path)
    
    df = df.drop(columns=['Id'])
    df = df.rename(columns={
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm': 'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm': 'petal_width'
    })
    
    print("Splitting dataset into train/test splits...")
    train, test = train_test_split(df, test_size=0.4, stratify=df['Species'], random_state=42)
    
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.Species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.Species
    
    print("Training the Decision Tree Classifier model...")
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    mod_dt.fit(X_train, y_train)
    
    predictions = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(predictions, y_test)
    print(f"Training Success!!! Accuracy: {accuracy:.3f}")
    
    model_local_path = "artifacts/model.joblib"
    joblib.dump(mod_dt, model_local_path)
    print(f"Model saved locally to {model_local_path}")

if __name__ == "__main__":
    train_model()
