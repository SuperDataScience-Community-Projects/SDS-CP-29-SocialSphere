import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset using the correct path
df = pd.read_csv(os.path.join(script_dir, 'ssma.csv'))
print(df.head())