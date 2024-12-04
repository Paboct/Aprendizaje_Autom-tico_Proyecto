from model import APS_Solver
import pandas as pd

aps = APS_Solver()
aps.train_model("train_students.csv")
aps.test_model("train_students.csv")