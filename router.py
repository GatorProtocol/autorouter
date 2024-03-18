import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load  # Import joblib functions

class Router:
    def __init__(self):
        self.data_file = "database/data.csv"
        self.model_file = "database/model.joblib"

        self.pipeline = self.load_model()
       
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            self.data = pd.DataFrame(columns=["query", "model"])
            self.data.to_csv(self.data_file, index=False)

    
    def load_model(self):
        try:
            return load(self.model_file)
        except FileNotFoundError:
            return make_pipeline(TfidfVectorizer(stop_words='english'), SGDClassifier(random_state=42))

    def save_model(self):
        dump(self.pipeline, self.model_file)


    def predict(self, query):
        if self.data.empty:
            raise ValueError("Model has not been trained. Add data and retrain.")
        return self.pipeline.predict([query])[0]

    def retrain(self):
        self.data = pd.read_csv(self.data_file)

        unique_labels = self.data["model"].unique()
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least two classes to train. Currently, there's insufficient variety in labels: {unique_labels}")

        X_train, X_test, y_train, y_test = train_test_split(self.data["query"], self.data["model"], test_size=0.25, random_state=42)
        self.pipeline.fit(X_train, y_train)
            
        self.save_model()

        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        open("database/accuracy.txt", "w").write(str(accuracy))

    def add(self, query, model):
        if str(model) in open("database/modellist.txt").read().split("\n"):
            new_data = pd.DataFrame([[query, model]], columns=["query", "model"])
            new_data.to_csv(self.data_file, mode='a', header=False, index=False)
        else:
            raise Exception("Invalid model ID")
