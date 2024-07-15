import streamlit as st    
import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# pipe1=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

def check():
    X=df[['Company', 'TypeName', 'Ram', 'Weight','TouchScreen', 'Ips',
       'ppi', 'Cpu_brand', 'HDD', 'SSD', 'Gpu brand',
       'os']]

    y=(df['Price'])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=2)


    step1=ColumnTransformer(transformers=[
        ('col_trf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')

    step2=RandomForestRegressor(n_estimators=110,
                            random_state=3,
                            max_samples=0.9,
                            max_features=2,
                            max_depth=16)

    pipe1=Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    pipe1.fit(X_train,y_train)

    # y_pred=pipe1.predict(X_test)
    return pipe1






st.title("Laptop Price Predictor")


#Brand name
company=st.selectbox('Brand',df['Company'].unique())

#type of laptop
type_n=st.selectbox('Type',df['TypeName'].unique())

# Ram
ram=st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
weight=st.number_input("Weight of the Laptop")

# TouchScreen
touchscreen=st.selectbox('TouchScreen',['No','Yes'])

# IPS Display
ips=st.selectbox('IPS',['No','Yes'])

# In PPI we need two things (Screenresolution and screesize)
# 1.Screensize
screen_size=st.number_input('Screen Size')

# Resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900',
                                        '3840x2160','3200x1800','2880x1800','2560x1600',
                                        '2560x1440','2304x1440'])

# CPU
cpu=st.selectbox('CPU',df['Cpu_brand'].unique())

#HDD
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# SSD
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#Gpu
gpu=st.selectbox('GPU',df['Gpu brand'].unique())
# OS system
os=st.selectbox('OS',df['os'].unique())


if st.button("Predict Button"):
    # Query
    ppi=None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0


    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    # ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    ppi=((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type_n,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    print("Data:",query)
    pipe1=check()
    st.title("The predicted price of Laptop  "   + str((pipe1.predict(query)[0]).round(0)))
    print("Orgial Data",query)
    

    
