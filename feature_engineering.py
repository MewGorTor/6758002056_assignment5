import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/train_validation.csv")

y = df["ProdTaken"].astype(int)
X = df.drop("ProdTaken", axis=1)

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, "data/preprocessor.joblib")

print("X shape:", X.shape)
print("Processed X shape:", X_processed.shape)
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
print("Target distribution:")
print(y.value_counts())