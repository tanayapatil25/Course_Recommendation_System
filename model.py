import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

# Assuming df is your dataset
df = pd.read_csv('Dataset_RS.csv')

# Drop unnecessary columns
df = df.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Student Name'], axis=1)

# Drop rows with null values in 'Graduation'
df = df.dropna(subset=['Graduation'])

# Drop rows where 'entrance test score' is less than 20
df = df[df['entrance test score'] >= 20]

# Separate features and target variable
X = df[['Graduation', 'Interest', 'prerequisites', 'entrance test score']]
y = df['Recommend Course']

# Apply label encoding to the target variable 'Recommend Course'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Combine X_train and X_test for consistent encoding
X_combined = pd.concat([X_train, X_test])

# Apply encoding using OneHotEncoder and LabelEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('graduation_interest_prerequisites', OneHotEncoder(handle_unknown='ignore'), ['Graduation', 'Interest', 'prerequisites']),
    ],
    remainder='passthrough'
)

# Fit the preprocessing on the combined dataset
preprocessor.fit(X_combined)

# Transform the training and testing data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define the XGBoost model
model = xgb.XGBClassifier()

# Create a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Print accuracy on the test set
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Save the entire pipeline, including preprocessor and model
joblib.dump(pipeline, 'your_pipeline.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')


