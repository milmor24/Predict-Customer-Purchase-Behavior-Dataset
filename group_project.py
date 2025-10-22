import pandas as pd

df = pd.read_csv("data.csv") 

# Remove decimals by converting to integers
df["AnnualIncome"] = df["AnnualIncome"].astype(int)
df["TimeSpentOnWebsite"] = df["TimeSpentOnWebsite"].astype(int)