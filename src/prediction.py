import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('../datasets/finaltraining_data.csv')

TARGET_COLS = ['Assigned Department', 'Priority', 'Sentiment', 'Action Type']
label_encoders = {}

for col in TARGET_COLS:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le


joblib.dump(label_encoders, 'models/encoders.pkl')



X = df['Customer Query']
Y1 = df[['Assigned Department_Encoded', 'Priority_Encoded', 'Sentiment_Encoded']]


X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size=0.2, random_state=42)

model_1_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultiOutputClassifier(LogisticRegression(solver='lbfgs', random_state=42)))
])

model_1_pipeline.fit(X_train, Y1_train)
joblib.dump(model_1_pipeline, 'models/triage_model.pkl')


### MOdel pipeline 2 

Y2 = df['Action Type_Encoded']


X_train_act, X_test_act, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=42)

model_2_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(solver='lbfgs', random_state=42))
])

model_2_pipeline.fit(X_train_act, Y2_train)
joblib.dump(model_2_pipeline, 'models/action_model.pkl')


# --- Model 3: Resolution Recommendation (Similarity-Based) ---
from sklearn.neighbors import NearestNeighbors

# For this agent, we will model Resolution_Steps based on the Customer Query
# We use the entire dataset for the knowledge base
X_rec = df['Customer Query']
Y_rec = df['Resolution_Steps']

# 1. Vectorize the entire knowledge base (KB)
tfidf_rec = TfidfVectorizer(stop_words='english', max_features=5000)
X_rec_vectorized = tfidf_rec.fit_transform(X_rec)

# 2. Train a Nearest Neighbors model on the vectorized KB
# 'metric=cosine' makes it a Content-Based Recommender using cosine similarity
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn_model.fit(X_rec_vectorized)
joblib.dump(nn_model, 'models/knn_model.pkl')


# --- Model 4: Resolution Time Estimation (Regression) ---
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define features for the regression model
REGRESSION_FEATURES = ['Complexity_Score', 'Priority_Encoded', 'Assigned Department_Encoded', 'Sentiment_Encoded']
TARGET_REG = 'Resolution_Time_Actual'

# Prepare the data
X_reg = df[REGRESSION_FEATURES]
Y_reg = df[TARGET_REG]

# Define preprocessor: Scale numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Complexity_Score']),
        ('cat', 'passthrough', ['Priority_Encoded', 'Assigned Department_Encoded', 'Sentiment_Encoded'])
    ],
    # remainder='passthrough'
)

X_train_reg, X_test_reg, Y2_reg_train, Y2_reg_test = train_test_split(X_reg, Y_reg, test_size=0.2, random_state=42)


# Build and train the regression pipeline
model_3_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()) # Linear Regression
])

model_3_pipeline.fit(X_train_reg, Y2_reg_train)

joblib.dump(model_3_pipeline, 'models/regre_model.pkl')


### Evaluation

from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import numpy as np

def evaluate_static_layer(y_true_dept, y_pred_dept, y_true_time, y_pred_time,y_true_act,y_pred_act):
    # Classification Performance (Department/Priority)
    
    hamming_loss = np.mean(np.not_equal(y_true_dept, y_pred_dept))
    accu = 1 - hamming_loss

    accuracy = accuracy_score(y_true_act,y_pred_act)
    # f1_scor = f1_score(y_true_act,y_pred_act,average='macro')
    # Regression Performance (Resolution Time)
    mae = mean_absolute_error(y_true_time, y_pred_time)
    print("Accuracy of multiclass-multioutput(Triage) is ", + round(accu,3))
    print("Accuracy of single classification (action) is ", + round(accuracy,3))
    # print("F1_score of single classification (action) is ", + round(f1_scor,3))
    print("classification report  (action) is \n ", classification_report(y_true_act,y_pred_act))

    print(f"Resolution Time MAE: {mae:.2f} minutes")
    
    return {"accuracy": accu, "mae": mae}

evaluate_static_layer(Y1_test,model_1_pipeline.predict(X_test),Y2_reg_test,model_3_pipeline.predict(X_test_reg),Y2_test,model_2_pipeline.predict(X_test_act))


#### Functions 

def resolution_recommender(query, n=3):
    """Predicts the top N most similar resolution steps."""
    query_vectorized = tfidf_rec.transform([query])
    # Find the N nearest neighbors (indices)
    distances, indices = nn_model.kneighbors(query_vectorized, n_neighbors=n)

    recommendations = []
    for i, index in enumerate(indices[0]):
        recommendations.append({
            'Similarity_Score': 1 - distances[0][i], # 1 - distance = similarity for cosine
            'Ticket_ID': df.iloc[index]['Ticket ID'],
            'Recommended_Steps': df.iloc[index]['Resolution_Steps']
        })
    return recommendations






def time_estimator(dept_enc, priority_enc, sentiment_enc, complexity):
    """Predicts Resolution Time in minutes."""
    input_data = pd.DataFrame([[complexity, priority_enc, dept_enc, sentiment_enc]],
                              columns=['Complexity_Score', 'Priority_Encoded', 'Assigned Department_Encoded', 'Sentiment_Encoded'])
    
    # Ensure time is positive (Check)
    predicted_time = model_3_pipeline.predict(input_data)[0]
    return max(0, predicted_time)



