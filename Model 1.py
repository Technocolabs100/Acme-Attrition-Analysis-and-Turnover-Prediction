from sklearn.linear_model import LogisticRegression
# Load data
X = dataset[['YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction']]
y = dataset['Attrition'].map({'Yes': 1, 'No': 0})
# Train the model
model = LogisticRegression()
model.fit(X, y)
# Predict attrition probability
dataset['AttritionProbability'] = model.predict_proba(X)[:, 1]
# Output the dataset with the new probability column
dataset

