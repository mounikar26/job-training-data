import pandas as pd
import numpy as np

# Set random seed so the data is the same every time
np.random.seed(42)

# Number of rows
n = 1000

# Generate covariates
age = np.random.randint(20, 60, size=n)
education = np.random.randint(10, 20, size=n)  # years of education
prior_income = np.random.normal(25000, 5000, size=n).round()
experience = (age - 18) + np.random.randint(-5, 5, size=n)

# Probability of being treated depends on education and income
prob_treated = 1 / (1 + np.exp(-0.2*(education - 14) - 0.0002*(prior_income - 25000)))
treatment = (np.random.rand(n) < prob_treated).astype(int)

# Earnings based on factors + treatment effect + randomness
treatment_effect = 5000
base_earnings = prior_income * 1.2 + education * 400 + experience * 300
noise = np.random.normal(0, 5000, size=n)
post_earnings = base_earnings + (treatment * treatment_effect) + noise

# Create DataFrame
df = pd.DataFrame({
    'treatment': treatment,
    'post_earnings': post_earnings.round(0),
    'age': age,
    'education': education,
    'prior_income': prior_income,
    'experience': experience
})

# Save to CSV (so you can reuse it later)
df.to_csv("job_training_data.csv", index=False)

# Show first few rows
df.head()


!pip install pandas scikit-learn

import pandas as pd

df = pd.read_csv("job_training_data.csv")
print(df)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("job_training_data.csv")

# Step 1: Select features and treatment column
covariates = ['age', 'education', 'prior_income', 'experience']
X = df[covariates]
y = df['treatment']

# Step 2: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Fit logistic regression with higher max_iter
try:
    logit = LogisticRegression(max_iter=2000, solver='lbfgs')
    logit.fit(X_scaled, y)

    # Step 4: Save propensity scores in the dataframe
    df['propensity_score'] = logit.predict_proba(X_scaled)[:, 1]
    print("Logistic regression ran successfully.")
    print(df[['treatment', 'propensity_score']].head())

except Exception as e:
    print("Error occurred:", e)

from sklearn.neighbors import NearestNeighbors

# Separate treated and untreated groups
treated = df[df['treatment'] == 1].copy()
control = df[df['treatment'] == 0].copy()

# Fit nearest neighbor model on control group based on propensity score
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])

# Find closest control for each treated unit
distances, indices = nn.kneighbors(treated[['propensity_score']])

# Get matched control observations
matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
matched_treated = treated.reset_index(drop=True)

# Combine matched treated and control into one dataframe
matched_df = pd.concat([matched_treated, matched_control])

# Estimate Average Treatment Effect on the Treated (ATT)
att = matched_treated['post_earnings'].values - matched_control['post_earnings'].values
print(" Estimated ATT (PSM):", round(att.mean(), 2))

# Create a long-format dataset
df_long = pd.DataFrame({
    'id': df.index.tolist() * 2,
    'treatment': df['treatment'].tolist() * 2,
    'time': [0]*len(df) + [1]*len(df),  # 0 = before, 1 = after
    'earnings': df['prior_income'].tolist() + df['post_earnings'].tolist()
})

import statsmodels.formula.api as smf

# Fit the DiD model
model = smf.ols('earnings ~ treatment * time', data=df_long).fit()
print(model.summary())

import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set a nice style
sns.set(style="whitegrid")

# Convert to long format for plotting
df_long = pd.DataFrame({
    'id': df.index.tolist() * 2,
    'treatment': df['treatment'].tolist() * 2,
    'time': ['Before']*len(df) + ['After']*len(df),
    'earnings': df['prior_income'].tolist() + df['post_earnings'].tolist()
})

plt.figure(figsize=(10,6))
sns.boxplot(data=df_long, x='time', y='earnings', hue='treatment')
plt.title('Earnings Before and After Training (by Treatment Group)')
plt.xlabel('Time')
plt.ylabel('Earnings')
plt.legend(title='Treatment (0=No, 1=Yes)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.kdeplot(data=df[df['treatment'] == 1], x='propensity_score', label='Treated', fill=True)
sns.kdeplot(data=df[df['treatment'] == 0], x='propensity_score', label='Control', fill=True)
plt.title('Propensity Score Distribution (Before Matching)')
plt.xlabel('Propensity Score')
plt.legend()
plt.tight_layout()
plt.show()


