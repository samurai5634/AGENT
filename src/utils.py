import joblib

class ModelInterface:
    def __init__(self):
        # The agents now pull the 'frozen' intelligence from these files
        self.triage = joblib.load('models/triage_model.pkl')
        self.knn = joblib.load('models/knn_model.pkl')
        self.action = joblib.load('models/action_model.pkl')
        self.timer = joblib.load('models/regre_model.pkl')
        self.encoders = joblib.load('models/encoders.pkl')

# Create one global instance to be used by all tools
fi = ModelInterface()



