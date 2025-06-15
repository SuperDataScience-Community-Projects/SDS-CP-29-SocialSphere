import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


import pandas as pd
students= pd.read_csv('Students Social Media Addiction.csv')

# Step 1: Count students per country
country_counts = students['Country'].value_counts()

# Step 2: Define a threshold
threshold = 20

# Step 3: Replace low-frequency countries with 'Other'
students['Country_Grouped'] = students['Country'].apply(
    lambda x: x if country_counts[x] >= threshold else 'Other'
)


numerical_features= ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
       'Mental_Health_Score', 'Conflicts_Over_Social_Media']
categorical_features= ['Gender', 'Academic_Level', 'Country_Grouped', 'Most_Used_Platform',
       'Affects_Academic_Performance', 'Relationship_Status']

target = 'Addicted_Score'
# Setup: rows and columns for subplot grid
num_plots = len(numerical_features)
cols = 2
rows = math.ceil(num_plots / cols)


# Create subplot layout
fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{i} vs {target}' for i in numerical_features])

# Add one boxplot per subplot
for idx, feature in enumerate(numerical_features):
    row = idx // cols + 1
    col = idx % cols + 1
    for val in students[target].unique():
        fig.add_trace(
            go.Box(
                y=students[students[target] == val][feature],
                name=str(val),
                boxmean='sd',
                marker_color=None,
                showlegend=(idx == 0)  # Only show legend in the first plot
            ),
            row=row,
            col=col
        )

# Customize layout
fig.update_layout(
    height=rows * 400,
    width=cols * 600,
    title_text="Boxplots of Numerical Features by Target",
    showlegend=True
)

fig.show()
