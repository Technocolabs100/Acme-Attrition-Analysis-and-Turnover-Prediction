import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assume 'dataset' is already loaded into Power BI as a pandas DataFrame

# Prepare your features (X) and target variable (y)
X = dataset[['YearsSinceLastPromotion', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany']]
y = dataset['Attrition'].map({'Yes': 1, 'No': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict attrition probability on the entire dataset
dataset['AttritionProbability'] = model.predict_proba(X)[:, 1]

# Predict on the test set to calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(
    dataset['YearsAtCompany'], 
    dataset['YearsSinceLastPromotion'], 
    dataset['JobSatisfaction'], 
    c=dataset['AttritionProbability'], 
    cmap='coolwarm', 
    s=100  # size of points
)

# Add color bar to indicate the Attrition Probability
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label('Attrition Probability', fontsize=13)

# Set labels
ax.set_xlabel('Years at Company', fontsize=13)
ax.set_ylabel('Years Since Last Promotion', fontsize=13)
ax.set_zlabel('Job Satisfaction', fontsize=13)

# Set title

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=10)

# Display accuracy on the plot
accuracy_text = f'Model Accuracy: {accuracy * 100:.2f}%'
plt.figtext(0.30, 0.85, accuracy_text, fontsize=16, bbox=dict(facecolor='lightgrey', alpha=0.5))

# Show plot
plt.show()
