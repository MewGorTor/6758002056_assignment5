import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/class5data_clean.csv")

train_val, test = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

train_val.to_csv("data/train_validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("train_validation:", train_val.shape)
print("test:", test.shape)