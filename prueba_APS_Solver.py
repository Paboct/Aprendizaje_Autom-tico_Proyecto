from APS_Solver import APS_Solver
import pandas as pd

data = pd.read_csv('train_students.csv')
train_csv = data.sample(frac=0.7, random_state=42)
data = data.drop(train_csv.index)
test_csv = data.copy()
train_csv.to_csv('train_students2.csv', index=False)
test_csv.to_csv('test_students2.csv', index=False)

aps = APS_Solver()
aps.train_model('train_students2.csv')
aps.test_model('test_students2.csv')