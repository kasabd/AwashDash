#This is customer Credit Worthiness Modeling using machine learning. It predicts wether a customer is eligible for a given credit based on different feature

#!/usr/bin/env python
# coding: utf-8
# In[1]:
import pathlib
import pandas as pd
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV


# In[51]:


import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import warnings


# In[54]:

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
df= pd.read_csv(DATA_PATH.joinpath('bankloans.csv'))


#rename columns to the proper name
df.rename(columns={"ed": "education", "employ": "exprience","debtinc":"dept2incomeratio","creddebt":"credit2deptratio"}, inplace=True)


# In[169]:


df.head(10)


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[170]:


#Fillna of Age with  with mean()
df['age']=df['age'].fillna(df['age'].mean())


# In[171]:


df['age'].mean()


# In[172]:


#Filter defaullt column with Null value
df_null = df.isnull().any(axis=1)
deflt_nul = df[df_null]


# In[173]:


deflt_nul.isnull().sum()


# In[174]:


#Now drop the null value in default
df.dropna(axis=0,inplace=True)


# In[175]:


#check null value after droping
df.isnull().sum()


# In[176]:


data=df.copy()


# In[177]:


#start training and testing preparation here
y=df[['default']].copy()


# In[178]:


x=df.drop(['default'],axis=1).copy()


# In[179]:


#Normalize usind standard scaler
col=['age','education','exprience','address','income','dept2incomeratio','credit2deptratio','othdebt']
scl=StandardScaler()


# In[180]:


#see after Normalizing 
df.head(5)


# In[181]:


xtrain, xtest, ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[182]:


#Normalize
xtrain[col]=scl.fit_transform(xtrain[col])


# In[183]:


logst=LogisticRegression()


# In[184]:


logst.fit(xtrain,ytrain)


# In[185]:


logst.score(xtest,ytest)


# In[186]:


logst.score(xtrain,ytrain)


# In[189]:


pred=logst.predict(xtest)


# In[190]:


result=xtest


# In[191]:


result['Actual']=ytest


# In[192]:


result['predicted']=pred


# In[193]:


result.head(20)


# In[47]:


df.columns


# In[46]:


#confussion Matrix of logistic regression prediction
confusion_matrix(ytest,pred)


# In[201]:


# Create the Dash app
# external_stylesheets = ['https://fonts.googleapis.com/css2?family=Open+Sans&display=swap']
app = dash.Dash(__name__)
server = app.server
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Define the layout of the dashboard
app.layout = html.Div(
#     style={'font-family': 'Open Sans'}, 
    children=[
    html.H1(' Customer Credit Scoring using LOGISTIC REGRESSION'),
    html.Div([
        html.H3('Exploratore relationship between variables'),
        html.Label('Feature 1 (X-axis)'),
        dcc.Dropdown(
            id='x_feature',
            options=[{'label': col, 'value': col} for col in data.columns],
            value=data.columns[0]
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Feature 2 (Y-axis)'),
        dcc.Dropdown(
            id='y_feature',
            options=[{'label': col, 'value': col} for col in data.columns],
            value=data.columns[1]
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    dcc.Graph(id='correlation_plot'),
    # Customer credit scoring based on predictors
    html.H3("Customer credit scoring "),
    html.Br(), 
    html.Br(),
    html.H3("Please Enter the predictors of customer credit scoring"),    
    html.Div([
        html.Label("Age:   "),
        dcc.Input(id='age', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Label("Education:   "),
        dcc.Input(id='edu', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Label("Work Exprience:   "),
        dcc.Input(id='exp', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Label("Address:   "),
        dcc.Input(id='add', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Label("Income:   "),
        dcc.Input(id='inc', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Label("Dept to income ratio:   "),
        dcc.Input(id='debtinc', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Label("Credit to debt ratio:   "),
        dcc.Input(id='creddebt', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Label("Other debts:   "),
        dcc.Input(id='otherdebt', type='number', required=True), 
        html.Br(),
    ]),
    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0), 
    ]),
    html.Div([
        html.H4("Predicted Customer rating"),html.Div(id='prediction-output')
    ])
])
@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value'),
    dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature)
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig
# Define the callback function to predict customer credi worthiness
@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('age', 'value'),
     State('edu', 'value'),
     State('exp', 'value'),
     State('add', 'value'),
     State('inc', 'value'),
     State('debtinc', 'value'),
     State('creddebt', 'value'),
     State('otherdebt', 'value')],
     prevent_initial_call=True    
)
def predict_CreditRisk(n_clicks,age,edu,exp,add,inc,debtinc,creddebt,otherdebt):
    # Create input features array for prediction
    input_features = np.array([age,edu,exp,add,inc,debtinc,creddebt,otherdebt]).reshape(1, -1)
    # Predict the customer credit worthiness
    prediction = logst.predict(input_features)[0]
    #return html.Div(id='prediction-output',children={})
    if prediction == 1:
        return 'The customer will likely default.'
    else:
        return 'The customer is eligible for credit.'
if __name__ == '__main__':
    app.run_server(port = 6099, debug=True,use_reloader=False)


# In[ ]:




