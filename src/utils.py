import pandas as pd
import joblib
import numpy as np
import chromadb

class ModelInterface:
    def __init__(self):
        # Load the frozen intelligence weights and configurations
        self.triage = joblib.load('models/triage_model.pkl')
        self.knn = joblib.load('models/knn_model.pkl')
        self.action = joblib.load('models/action_model.pkl')
        self.timer = joblib.load('models/regre_model.pkl')
        self.encoders = joblib.load('models/encoders.pkl')
        self.tfidf_rec = joblib.load('models/tfidf_rec.pkl')
        self.df = pd.read_csv('../datasets/finaltraining_data.csv')

        

   
# Create one global instance to be exposed to your tools and app layer
fi = ModelInterface()

