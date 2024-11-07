#Data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train_students.csv')
print(f"Data:\n{data.head()}\n")
print(f"Stadistical data:\n{data.describe()}\n")