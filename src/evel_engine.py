import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix

class RealAnalyticsEngine:
    def __init__(self, csv_path=("../datasets/finaltraining_data.csv")):
        self.csv_path = csv_path
        
        # Load your exact binary models and encoder pipeline mappings
        self.triage_model = joblib.load("models/triage_model.pkl")
        self.regre_model = joblib.load("models/regre_model.pkl")
        self.action_model = joblib.load("models/action_model.pkl")
        self.knn_model = joblib.load("models/knn_model.pkl")
        self.encoders = joblib.load("models/encoders.pkl")
        
        # Load the data frame to calculate tracking evaluation parameters
        self.df = pd.read_csv(self.csv_path)
        self.generate_live_metrics()

    def generate_live_metrics(self):
        """Applies encoders to raw headers to build exact matching validation matrices."""
        processed_df = self.df.copy()
        
        # 1. ENCODE ATTRIBUTES LIVE USING ENCODERS.PKL MAPPINGS
        try:
            # Safely mapping the label encoders precisely as structured in your pipeline
            processed_df['Priority_Encoded'] = self.encoders['Priority'].transform(processed_df['Priority'])
            processed_df['Assigned Department_Encoded'] = self.encoders['Assigned Department'].transform(processed_df['Assigned Department'])
            processed_df['Sentiment_Encoded'] = self.encoders['Sentiment'].transform(processed_df['Sentiment'])
            processed_df['Action Type_Encoded'] = self.encoders['Action Type'].transform(processed_df['Action Type'])
        except Exception as e:
            print(f"Encoding Error: Mismatch between raw CSV names and encoders.pkl. Details: {e}")
            return

        # 2. REGRESSION FEATURE EXTRACTION: Exactly ['Priority_Encoded', 'Assigned Department_Encoded', 'Sentiment_Encoded']
        # Building the 2D feature matrix your regre_model expects
        X_reg = processed_df[['Priority_Encoded', 'Assigned Department_Encoded', 'Sentiment_Encoded', 'Complexity_Score']]
        
        # Check if your CSV has 'Actual_Resolution_Time' or use the last column as y target
        y_reg_true = self.df['Actual_Resolution_Time'] if 'Actual_Resolution_Time' in self.df.columns else self.df.iloc[:, -1]
        
        self.y_reg_pred = self.regre_model.predict(X_reg)
        self.r2 = r2_score(y_reg_true, self.y_reg_pred)
        self.mae = mean_absolute_error(y_reg_true, self.y_reg_pred)
        
      
        X_clf = self.df['Customer Query']
        
        
        # Ground truth values for categorization evaluation
        y_clf_true = self.df['Action Type']
        y_clf_pred = self.action_model.predict(X_clf)

        print("X_clf shape:", X_clf.shape)
        print("y_clf_true shape:", y_clf_true.shape)

        try:
            print("y_clf_pred shape:", y_clf_pred.shape)
        except:
            print("y_clf_pred:", y_clf_pred)

        print("type:", type(y_clf_pred))

        self.clf_accuracy = accuracy_score(y_clf_true, y_clf_pred)
        
        # 4. CRITICAL OVERRIDE TRACKING
        # True breach boundary condition from your notebook logic (> 240 mins)
        self.df['True_Breach'] = np.where(y_reg_true > 240, 1, 0)
        self.df['Predicted_Breach'] = np.where(self.y_reg_pred > 240, 1, 0)

    def get_dashboard_metrics(self):
        return {
            "r2": f"{self.r2:.3f}",
            "mae": f"{self.mae:.1f} mins",
            "clf_acc": f"{self.clf_accuracy * 100:.1f}%",
            "df_eval": self.df
        }

    def get_override_confusion_matrix(self):
        return confusion_matrix(self.df['True_Breach'], self.df['Predicted_Breach'])