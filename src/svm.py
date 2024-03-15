#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV


# In[48]:


import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import warnings


# In[106]:


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
df= pd.read_csv(DATA_PATH.joinpath('bankloans.csv'))



# In[107]:


df.head(5)


# In[108]:


#rename columns to the proper name
df.rename(columns={"ed": "education", "employ": "exprience","debtinc":"dept2incomeratio","creddebt":"credit2deptratio"}, inplace=True)


# In[109]:


df.head(10)


# In[110]:


df.columns


# In[111]:


df.shape


# In[112]:


df.dtypes


# In[113]:


df.describe()


# In[114]:


df.isnull().sum()


# In[12]:


df.isnull().sum()


# In[115]:


#Fillna with mean()
df['age']=df['age'].fillna(df['age'].mean())


# In[116]:


df['age'].mean()


# In[117]:


#after filling  null age with mean
df.isnull().sum()


# In[118]:


#Filter defaullt column with Null value
df_null = df.isnull().any(axis=1)
deflt_nul = df[df_null]


# In[119]:


#Remove  default null value  after filtering it as deflt_nul above
df.dropna(axis=0,inplace=True)


# In[120]:


df.isnull().sum()


# In[121]:


data=df.copy()


# In[122]:


data.head()


# In[228]:


#start training and testing preparatio
y=df[['default']].copy()


# In[232]:


x=df.drop(['default'],axis=1).copy()


# In[233]:


col=['age','education','exprience','address','income','dept2incomeratio','credit2deptratio','othdebt']
scl=StandardScaler()


# In[234]:


xtrain, xtest, ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[188]:


#Normalize
#xtrain[col]=scl.fit_transform(xtrain[col])


# In[ ]:


#SUPPORT VECTOR MACHINE: Credit Risk prediction using SUPPORT VECTOR MACHIN


# In[235]:


SVM=SVC()


# In[243]:


model=SVM.fit(xtrain,ytrain)


# In[244]:


#training accuracy of the support vector machine
model.score(xtrain,ytrain)


# In[245]:


#out of sample prediction accuracy of the support vector machine with out cross validation
model.score(xtest,ytest)


# In[ ]:





# In[253]:


# Create the Dash app
# external_stylesheets = ['https://fonts.googleapis.com/css2?family=Open+Sans&display=swap']

app = dash.Dash(__name__)
server = app.server
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout of the dashboard
app.layout = html.Div(
#     style={'font-family': 'Open Sans'}, 
    children=[
    
    html.H1(' Customer Credit Scoring using SUPPORT VECTOR MACHINE'),
    
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
        html.Br(),
        html.Label("Education:   "),
        dcc.Input(id='edu', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Work Exprience:   "),
        dcc.Input(id='exp', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Address:   "),
        dcc.Input(id='add', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Income:   "),
        dcc.Input(id='inc', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Dept to income ratio:   "),
        dcc.Input(id='debtinc', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Credit to debt ratio:   "),
        dcc.Input(id='creddebt', type='number', required=True),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label("Other debts:   "),
        dcc.Input(id='otherdebt', type='number', required=True), 
        html.Br(),
        html.Br(),
        html.Br(),
    ]),

    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0),
        
    ]),

    html.Div([
        html.H4("Predicted Customer rating"),
        html.Div(id='prediction-output')
    ])
])

# Define the callback to update the correlation plot
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
def predict_creditscore(n_clicks,age,edu,exp,add,inc,debtinc,creddebt,otherdebt):
    # Create input features array for prediction
    input_features = np.array([age,edu,exp,add,inc,debtinc,creddebt,otherdebt]).reshape(1, -1)

    # Predict the customer credit worthiness
    prediction = model.predict(input_features)[0]
   
    if prediction == 1:
        return 'The customer will likely default.'
    else:
        return 'The customer is eligible for credit.'
if __name__ == '__main__':
    app.run_server(port = 7026, debug=True,use_reloader=False)


# In[247]:


pred=model.predict(xtest)


# In[248]:


xtestnorm=xtest.copy()


# In[249]:


xtestnorm['actual']=ytest


# In[250]:


xtestnorm['predicted2']=pred


# In[252]:


xtestnorm.head(50)


# In[167]:


len(xtest)


# In[ ]:




