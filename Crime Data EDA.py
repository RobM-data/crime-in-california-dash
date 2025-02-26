#!/usr/bin/env python
# coding: utf-8

# # Notebook Introduction

# The data has already been cleaned and in this notebook, I perform some basic EDA to see trends in crime and identify potential relationships between some of the variables. By getting a deeper understanding of the data, I will be able to make more informative decisions about what I want to include in the Tableau dashboard.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("Crime Data Cleaned.csv")


# In[5]:


df = df.drop("Unnamed: 0", axis=1)
df.head()


# In[6]:


df.dtypes


# Some of the datatypes are not identified properly so I'll correct them.

# In[8]:


df["Date_Reported"] = pd.to_datetime(df["Date_Reported"])
df["Date_Occurred"] = pd.to_datetime(df["Date_Occurred"])
df["Victim_Age"] = df["Victim_Age"].astype("Int64")


# # Trends over Time

# I think one of the most interesting things to investigate will be how crime in California has changed over time. This could include the volume of crime, which types of crime are committed, and how the demographics of the victims have changed.

# ### Has the total volume of crime changed over time?

# In[12]:


data = df["Date_Occurred"].groupby(df["Date_Occurred"].dt.to_period("W")).agg("count")
data.plot()
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.title("Change in Crime Volume over Time")
plt.show()


# According to this graph, the number of crimes committed stayed relatively constant until the start of 2021, then increased until the start of 2023. From there it had a tiny downward trend, dropping suddenly just before 2024. According to the data source, LAPD started facing issues with uploading their data on 2024/01/18. This means that although all data after this date is still accurate, the quantity of data is not. So when dealing with counts of data, anything after this date should not be included.

# In[14]:


data = data[data.index < "2023-01-01"]
data.plot()
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.title("Change in Crime Volume over Time")
plt.show()


# Now we have a graph that makes more sense. Using a time series model like SARIMA, the future volume of crimes can be predicted.

# In[16]:


from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
data.index = data.index.to_timestamp()


# In[17]:


auto_sarima = auto_arima(
    data,
    seasonal=True,
    m=52,
    stepwise=True, 
    suppress_warnings=True,
)

order = auto_sarima.order
seasonal_order = auto_sarima.seasonal_order

sarima_model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit()

forecast = sarima_result.forecast(100)


# In[18]:


data.plot()
forecast.plot()
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.title("Change in Crime Volume over Time")


# According to this forecast, the number of crimes will continue to steadily rise. The model hasn't been tuned at all but this gives a good idea of the trend of crimes over time.

# ### Have the types of crime committed changed over time?

# I want to create a stacked bar chart that shows the 5 most common crimes for each quarter before 2024.

# In[22]:


import plotly.express as px

data = df.groupby(df["Date_Occurred"].dt.to_period("Q"))["Crime_Desc"].value_counts().groupby(level=0, group_keys=False).head(5)
data = data.reset_index()
data.rename(columns={"Date_Occurred": "Year", "Crime_Desc": "Crime Type", "count": "Occurrences"}, inplace=True)

data = data[data["Year"] < "2024"]

data["Year"] = data["Year"].astype(str)

fig1 = px.bar(
    data,
    x="Year",
    y="Occurrences",
    color="Crime Type",
    title="Most Common Crimes by Year", 
    labels=False,
    color_discrete_sequence=px.colors.qualitative.Set1,
    height=530
)

fig1.update_layout(
    yaxis_title=None, 
    xaxis_title=None, 
    yaxis=dict(showgrid=False),
    xaxis=dict(showgrid=False),
    title=dict(x=0.42, y=0.87)
)
fig1.show()


# This stacked bar chart shows a lot of consistency in the types of crime committed year by year. "Stolen Vehicle" is almost always the most frequent crime, with "Simple Assault" normally second. "Burglary From Vehicle", "Burglary" and "Felony Vandalism" also appear regularly. The only change that seems to have happened was a short period from 2022 to early 2023 when "Identity Theft" became very common, even surpassing "Stolen Vehicle" in 2022 Q4.

# ### Which demographic is most likely to be the victim of a crime?

# There are three main columns that can help describe the demographic of the victim: gender, age and ethnicity. I'll compare the gender statistics just with numbers, and then create a histogram for age and bar chart for ethnicity. This will help give a general overview of which kinds of people are more likely to be a victim of crime. Then, I'll create a treemap to show all of this information in one go.

# In[26]:


df["Victim_Gender"].value_counts()


# Slightly more males are victims of crime than females. Actually it would be nice to see if there's any difference in the kinds of crime that males and females experience.

# In[28]:


data = df.groupby(df["Victim_Gender"])["Crime_Desc"].value_counts().groupby(level=0, group_keys=False).head(7)
data = data.reset_index()
data.rename(columns={"Victim_Gender": "Gender", "Crime_Desc": "Crime Type", "count": "Occurrences"}, inplace=True)

fig2 = px.bar(
    data,
    x="Gender",
    y="Occurrences",
    text="Crime Type",
    color="Crime Type",
    title="Types of Crime Experienced by Each Gender", 
    labels=False,
    color_discrete_sequence=px.colors.qualitative.Set1,
    height=530
)

fig2.update_layout(
    yaxis_title=None, 
    xaxis_title=None, 
    yaxis=dict(showgrid=False),
    xaxis=dict(showgrid=False),
    title=dict(x=0.5, y=0.87),
    showlegend=False
)
fig2.show()


# There is quite a lot of similarity between the kinds of crime that males and females experience. In fact, they share 6 of the top 7 crimes. There is just one crime that stands out for either gender. Females are more likely to be th victim of assault by their partner, and males are much more likely to be the victim of assault with a deadly weapon.

# In[30]:


fig3 = px.histogram(
    df, 
    x="Victim_Age",
    title="Distribution of Victim Ages",
    height=450
)

fig3.update_layout(
    yaxis_title=None, 
    xaxis_title=None, 
    yaxis=dict(showgrid=False),
    xaxis=dict(showgrid=False),
    title=dict(x=0.5, y=0.85)
)
fig3.show()


# The victim ages follow a fairly smooth distribution with the most frequent age being 30. The distribution is left skewed, meaning the victim ages are more concentrated around lower values, and higher ages are more rare. Also, the frequency only increases slowly to start off, and then increases suddenly around 18, meaning that people are not as willing to commit crimes on children.

# In[32]:


data = df.groupby("Victim_Ethnicity").agg("count").reset_index().sort_values("Date_Occurred")
res = data[data["Date_Reported"] <= 4000]["Date_Reported"].sum()
data[data["Victim_Ethnicity"] == "Other"]["Date_Reported"] = data[data["Victim_Ethnicity"]=="Other"]["Date_Reported"] + res
data = data[data["Date_Reported"] > 4000]
data = data[data["Victim_Ethnicity"] != "Unknown"]


# In[33]:


colorscale = [
    [0.0, "rgb(100, 149, 237)"],  
    [1.0, "rgb(0, 0, 139)"]       
]

fig4 = px.bar(
    data,
    x="Victim_Ethnicity",
    y="Date_Reported",
    title="Ethnicity of Victims",
    height=440,
    color="Date_Reported",
    color_continuous_scale=colorscale
)

fig4.update_layout(
    yaxis_title=None, 
    xaxis_title=None, 
    yaxis=dict(showgrid=False),
    xaxis=dict(showgrid=False),
    title=dict(x=0.5, y=0.84),
)
fig4.update_coloraxes(showscale=False)
fig4.show()


# This bar chart shows that Hispanic people are most often the victims of crime, with White and Black people also being considerably high. Other ethnicities such as Chinese, Filipino and Korean are much less often victims of crime, but for a proper analysis this should be compared with the total population of these ethnicities in California. It is probable that the ethnicities with lower frequences on this graph are also those with a lower total population. 

# In[35]:


age_bins = list(range(0, 101, 10))
age_labels = [f"{i}-{i+9}" for i in age_bins[:-1]]
df["Age_Group"] = pd.cut(df["Victim_Age"], bins=age_bins, labels=age_labels, right=False)

data = df.value_counts(subset=["Victim_Gender", "Age_Group", "Victim_Ethnicity"]).reset_index()
data = data.query("Victim_Ethnicity not in ['Other', 'Unknown']")
data = data.query("count > 5000")


# In[36]:


fig5 = px.treemap(
    data,
    path=["Victim_Gender", "Victim_Ethnicity", "Age_Group"],
    values="count",
    color="Age_Group",
    title="Victim Demographics by Gender, Ethnicity and Age",
    branchvalues="total",
    height=580
)

fig5.update_traces(
    textinfo="label+value",
    textfont=dict(
        family="Arial", 
        size=14, 
        color="black"
    )
)
fig5.update_layout(
    title=dict(x=0.5, y=0.9)
)
fig5.show()


# This tree map shows demographics which have over 5000 occurences in the data. We can see that victims are spread almost evenly across male and female, and in both cases Hispanic, White and Black are the top three victims. Even within gender and ethnicity, the top age groups remains relatively constant, 20-29 and 30-39 having the highest frequencies in a lot of cases. 40-49 and 50-59 tend to come next, with 60-69 being last. The 70-79 age group appears once in the whole map, in the Male White section, and the 60-69 age group in this section is also pretty large, so this could be worth investigating more. Also, the 10-19 age group only appears for the Hispanic ethnicity, which is another interesting feature of the data. There could be many reasons why older White people and younger Hispanic people are more often the victims of crime.

# ### What times do crimes normally occur?

# ### Where do crimes normally occur?

# # Summary

# In[41]:


import dash
from dash import dcc, html


# In[42]:


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("My Interactive Dashboard"),
    dcc.Graph(figure=fig1),  # Embed Plotly figure
    dcc.Graph(figure=fig2),  # Embed Plotly figure
    dcc.Graph(figure=fig3),  # Embed Plotly figure
    dcc.Graph(figure=fig4),  # Embed Plotly figure
    dcc.Graph(figure=fig5)  # Embed Plotly figure
])

if __name__ == "__main__":
    app.run_server(debug=True)


# In[ ]:




