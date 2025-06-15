import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


# Set page config
st.set_page_config(layout="wide", page_title="Student Social Media Addiction Analysis")

# Title
st.title("Exploratory Data Analysis - Student Social Media Addiction Dataset")

st.write("""This dataset explores how social media usage affects students across different countries, 
         platforms, and academic performance metrics. The analysis focuses on patterns in addiction, 
         mental health, and academic impact.""")
# Load data
@st.cache_data
def load_data():
    return pd.read_csv("submissions/team-members/aditi-phadnis/Students Social Media Addiction.csv")

students = load_data()


# Data preprocessing
country_counts = students['Country'].value_counts()
threshold = 10
students['Country_Grouped'] = students['Country'].apply(lambda x: x if country_counts[x] >= threshold else 'Other')

# Sidebar filters
st.sidebar.header("Filters")
selected_gender = st.sidebar.multiselect("Gender", options=students['Gender'].unique(), default=students['Gender'].unique())
selected_level = st.sidebar.multiselect("Academic Level", options=students['Academic_Level'].unique(), default=students['Academic_Level'].unique())

# Apply filters
filtered_data = students[
    (students['Gender'].isin(selected_gender)) & 
    (students['Academic_Level'].isin(selected_level))
]

# Display filtered data
st.header("Data Overview")
st.write(filtered_data.head())
st.divider()

# Visualizations
st.header("Data Visualizations")


# Categorical features
st.subheader("Dataset Distribution")
cat_cols = ['Gender', 'Academic_Level', 'Country_Grouped', 'Most_Used_Platform', 
            'Affects_Academic_Performance', 'Relationship_Status']

fig = make_subplots(rows=3, cols=2, subplot_titles=cat_cols)

for i, col in enumerate(cat_cols):
    counts = filtered_data[col].value_counts()
    total = counts.sum()
    percentages = (counts / total) * 100

    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=percentages,
            name=col,
            hovertemplate='%{x}: %{customdata[0]} responses (%{y:.1f}%)<extra></extra>',
            customdata=np.stack((counts.values,), axis=-1),
            marker=dict(opacity=0.8)
        ),
        row=(i // 2) + 1,
        col=(i % 2) + 1
    )


fig.update_layout(height=1200, width=1000, title_text="Distribution of Categorical Features", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.write("""
## 📊 Analysis of characteristics of Dataset 
         
**Gender Ratio:**
The dataset has a balanced representation of both males and females, with no significant gender disparity.


**Academic Level:**
The dataset mainly consists of students at the **undergraduate level (325)**, followed by **graduate level (325)**, with a few students in **high school (27)**.

**Country:**
The dataset mainly includes students from **India (53)**, followed by the **United States (40)** and **Canada (34)**.  
Countries with very low representation have been grouped under **'Other'**.

**Most Used Platform:**
The top platforms are **Instagram (249)**, **TikTok (144)**, and **Facebook (123)**.  
**YouTube** is the least used platform with only **10** users.

**Affects Academic Performance:**
Around **64.3%** of respondents say **social media does not affect** their academic performance,  
while **35.7%** say **it does affect** their performance.

**Relationship Status:**
About **54.5%** of students are **single**, **41%** are **in a relationship**, and **4.5%** say **"it's complicated."**

         """)
st.divider()


# Numerical features vs Addicted Score
st.subheader("📊 Boxplots of Numerical Features by Addicted Score")
st.write("""
        The visual below presents **boxplots** of selected numerical features plotted against 
        the target variable: `Addicted_Score`.
""")
num_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']

fig = make_subplots(rows=2, cols=2, subplot_titles=num_cols)
target = 'Addicted_Score'

# Subplot grid dimensions
num_plots = len(num_cols)
cols = 2
rows = math.ceil(num_plots / cols)

# Create subplot figure
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f'{feature} vs {target}' for feature in num_cols]
)

# Add boxplots
for idx, feature in enumerate(num_cols):
    row = idx // cols + 1
    col = idx % cols + 1

    for val in sorted(students[target].unique()):
        fig.add_trace(
            go.Box(
                y=students[students[target] == val][feature],
                name=str(val),
                boxmean='sd',
                showlegend=(idx == 0)
            ),
            row=row,
            col=col
        )

    fig.update_yaxes(title_text=feature, row=row, col=col)
    fig.update_xaxes(
        title_text=target,
        row=row,
        col=col,
        categoryorder='array',
        categoryarray=sorted(students[target].unique())
    )

# Final layout settings
fig.update_layout(
    height=rows * 400,
    width=cols * 600,
    title_text="Boxplots of Numerical Features by Addicted Score",
    showlegend=True
)


# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.write(""" 

### 🔍 Feature-wise Observations

- **Age vs Addicted Score**  
  Age does not show a clear increasing or decreasing trend with addiction scores. The spread is fairly uniform, with some minor variations and outliers across the range.

- **Average Daily Usage Hours vs Addicted Score**  
  A clear positive trend is evident — students with higher addiction scores tend to have significantly higher average daily usage hours. This could suggest a strong behavioral link between screen time and addiction.

- **Sleep Hours Per Night vs Addicted Score**  
  There appears to be a mild inverse relationship. Higher `Addicted_Score` groups tend to have lower median sleep hours, though variability exists. This may indicate that addiction affects or coincides with poor sleep habits.

- **Mental Health Score vs Addicted Score**  
  Mental health scores tend to decline as addiction scores increase. The median mental health score is lower in higher addiction groups, suggesting a potential negative impact of addiction on mental well-being.

- **Conflict Over Social Media vs Addicted Score**  
  Conflicts over social media tend to rise as addiction scores increase. The median for number of conflicts over social media is higher in higher addiction groups, suggesting a potential negative impact of addiction on social relationships.
       
### 📌 Notes

- Each subplot shows the distribution of a numerical feature for each unique `Addicted_Score` level.
- Boxes represent the interquartile range (IQR), with medians and outliers clearly visible.
- Distinct colors are used for each group to aid visual separation.

         """)
st.divider()

st.subheader("Daily Usage vs Mental Health Score (colored by Academic Performance Impact")


fig2 = px.scatter(
    students,
    x='Avg_Daily_Usage_Hours',
    y='Mental_Health_Score',
    color='Affects_Academic_Performance',
    color_discrete_map={
        'Yes': 'red',
        'No': 'blue'
    },
    size='Addicted_Score',
    hover_data=['Country_Grouped', 'Gender'],
    title='Daily Usage vs Mental Health Score (colored by Academic Performance Impact)'
)

st.plotly_chart(fig2, use_container_width=True)

st.write("""
### 📊 Key Insights
1. **Inverse Relationship Between Metrics**  
Our analysis reveals a clear inverse correlation between addiction scores and mental health 
scores - as addiction levels increase, mental health scores tend to decrease.

2. **Academic Performance Impact**
The visualization demonstrates that students with lower mental health scores (associated with higher 
addiction levels) experience more significant negative effects on their academic performance.

3. **Visual Clarity**
We've employed distinct color coding for each group to enhance visual differentiation and facilitate 
clearer pattern recognition in the data.

         """)
st.divider()

st.write("""### 📊 Analysis of Categorical Features""")
categorical_features = ['Gender', 'Academic_Level', 'Country_Grouped', 'Most_Used_Platform',
                        'Affects_Academic_Performance', 'Relationship_Status']
target = 'Addicted_Score'

num_plots = len(categorical_features)
cols = 2
rows = math.ceil(num_plots / cols)

# Create subplot grid
fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f'{feature} vs {target}' for feature in categorical_features],
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for idx, feature in enumerate(categorical_features):
    row = idx // cols + 1
    col = idx % cols + 1

    # Count plot equivalent using Plotly
    count_df = students.groupby([feature, target]).size().reset_index(name='Count')

    fig.add_trace(
        go.Bar(
            x=count_df[feature],
            y=count_df['Count'],
            name=feature,
            marker=dict(color=count_df[target], colorscale='Viridis'),
            customdata=count_df[[target]],
            hovertemplate=f"{feature}: %{{x}}<br>{target}: %{{customdata[0]}}<br>Count: %{{y}}<extra></extra>"
        ),
        row=row,
        col=col
    )

# Final layout settings
fig.update_layout(
    height=rows * 400,
    width=1000,
    title_text='Categorical Features vs Addicted Score (Interactive)',
    barmode='group',
    showlegend=False
)

# Show plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Calculate average addicted score by gender
gender_scores = students.groupby('Gender')['Addicted_Score'].mean().round(2)


st.markdown("""
### 📊 Key Insights



1. **Gender Difference**  
   - Females have a slightly higher average addiction score (**6.52**) compared to males (**6.36**), with a minimal difference of **0.16 points**.  
   - This suggests that, on average, females in this dataset exhibit **marginally higher addiction levels** than males, but the gap is not substantial.



2. **Academic Level**  
   - Undergraduates have a slightly higher average addiction score (**6.49**) compared to graduates (**6.24**), a small difference of **0.25 points**.  
   - This indicates that **academic level alone is not a strong predictor** of addiction risk in higher education populations.



3. **Country Grouped**  
   - **India and the USA** have the highest average addiction scores.  
   - Further breakdown and analysis are provided in the heatmap section.


4. **Most Used Platform**  
   As discussed earlier, **Instagram, TikTok, and Facebook** emerged as the most widely used platforms among participants.

   - **Instagram**  
     - Highest addiction levels.  
     - **30 users** rated their addiction at **9** (on a scale where higher scores indicate greater addiction).

   - **TikTok**  
     - Second-highest addiction levels.  
     - **20 users** rated their addiction at **9**.

   - **Facebook**  
     - Lower addiction levels.  
     - Only **3 users** rated their addiction at **9**, while the **majority (50 users)** reported a moderate addiction score of **5**.

5. **Affects Acedemic Performance**
   -Students who have rated themselves 6 or above on addiction score mostly also  stated that use of social media has 
    affected their academic performance. 
6. **Relationship Status ❤️**

    | Relationship Status | Respondent Count | Average Addiction Score |
    |---------------------|------------------|--------------------------|
    | Complicated         | 32               | 7.03                     |
    | In Relationship     | 289              | 6.34                     |
    | Single              | 384              | 6.46                     |

- Students who marked their relationship status as **"Complicated"** have the **highest average addiction score (7.03)** among all groups.
- Those who are **"Single"** have a slightly higher addiction score (6.46) than those **"In a Relationship"** (6.34), but the difference is minor.
- The data suggests that individuals in **emotionally uncertain or unstable relationships** (e.g., complicated) may be more prone to higher social media addiction levels.
- However, while there are differences, they are **not drastic**, so relationship status may play a **moderate role** in influencing addiction behavior.


""")

st.divider()

# Create pivot table for heatmap
pivot = students.pivot_table(
    values='Addicted_Score',
    index='Country_Grouped',
    columns='Most_Used_Platform',
    aggfunc='mean'
).round(2)

import streamlit as st
import plotly.express as px

# Assuming 'pivot' is your pivot table (e.g., Country vs. Platform with Avg Addiction Score)
fig3 = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    title='Average Addiction Score by Country and Platform',
    color_continuous_scale='YlOrRd',
    labels=dict(x="Social Media Platform", y="Country", color="Avg Addiction Score")
)

# Adjust layout
fig3.update_layout(
    height=700,
    width=1000,
    title_font_size=22
)

# Display in Streamlit
st.write("### 🌍 Platform Usage Across Countries with addiction scores")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🔍 Key Observations from the Heatmap (Filtered for High Addiction: Score > 7)")
st.markdown(
""" 

#### 🟥 High Addiction Hotspots
- **India** shows very high addiction to:
  - **TikTok**: `8.60`
  - **Whatsapp**: `8.00`
&nbsp;
- **UK** shows very high addiction to:
  - **Twitter**: `8.00`
&nbsp;
- **USA** shows very high addiction to:
  - **TikTok**: `9.00`
  - **Instagram**: `8.68`
&nbsp;
- **Mexico** stands out with:
  - **Instagram**: `9.00`
  - **TikTok**: `8.00`
  - **Facebook**: `8.00`
&nbsp;
- **Russia** also reports high addiction to:
  - **Instagram**: `8.75`

### 📱 Top Addictive Platforms
- **TikTok** appears in **11 out of 26** high-addiction rows, making it the **most frequently high-scoring platform**. High addiction seen in:
  - India: `8.60`
  - Mexico: `8.00`
  - Maldives: `8.00`
  - Bangladesh: `8.00`
  - Poland: `8.00`
  - Spain: `7.31`
  - Turkey: `7.31`
  - Nepal: `7.14`
  - UK: `7.67`
  - Italy: `7.25`
  - Other: `7.28`
&nbsp;
- **Instagram** appears in **9 rows**, showing widespread addiction:
  - Russia: `8.75`
  - USA: `8.68`
  - Bangladesh: `7.17`
  - India: `7.22`
  - Turkey: `7.36`
  - Pakistan: `7.44`
  - Poland: `8.00`
  - Mexico: `9.00`
  - Other: not listed, but high in grouped regions
&nbsp;
- **Facebook** shows high addiction in:
  - Spain: `8.00`
  - Mexico: `8.00`
  - UK: `7.25`
  - Bangladesh: `8.00`

### 🌍 “Other” Countries
Even in the **"Other" grouped countries** (countries with low individual sample size), some platforms report high addiction:
- **TikTok**: `7.28`
- **Snapchat**: `7.50`

This implies these platforms have a **consistently high addictive effect globally**, not just in dominant countries.

### 📉 Low Addiction Platforms
Platforms like:
- **LINE**
- **KakaoTalk**
- **VKontakte**
- **WeChat**
- **LinkedIn**

do **not appear** in this high-addiction subset — indicating **low engagement or limited geographic use**, and thus **lower addictive tendencies** in the sample.

---

📌 *Conclusion*: **TikTok**, **Instagram**, and to some extent **Facebook** stand out as **globally addictive platforms**, with scores exceeding `8` in multiple countries.

"""

)


