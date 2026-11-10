
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
columns = [
    'checking_account', 'duration', 'credit_history',
    'purpose', 'credit_amount', 'savings',
    'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence', 'property',
    'age', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'dependents',
    'telephone', 'foreign_worker', 'credit_risk'
]

df = pd.read_csv(
    'german.data',
    sep=r'\s+',
    header=None,
    names=columns
)

# Prepare target
df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})

X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing
categorical_features = X.select_dtypes(
    include='object'
).columns

numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'),
     categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_accuracy = 0
best_model = None
best_name = None

for name, model in models.items():
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, predictions)

    print(name, ':', round(accuracy * 100, 2), '%')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# Save model and preprocessor
joblib.dump(best_model, 'credit_scoring_model.pkl')
joblib.dump(preprocessor, 'credit_preprocessor.pkl')

print("\nTraining completed!")
print("Best Model:", best_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Model and preprocessor saved successfully!")
