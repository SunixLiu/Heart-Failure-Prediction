import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly_express as px
import streamlit.components.v1 as components

import shap
import pickle
import time

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
   page_title="CVD patient heartfailure survival and hazard prediction",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

#预测生存概率
def predict_survival(model, df):
    predicted_data= model.predict_survival_function(df,return_array=True)
    return predicted_data

#预测累积发病风险
def predict_hazard(mode,df):
    predicted_data = model.predict_cumulative_hazard_function(df,return_array=True)
    return predicted_data

#加载训练好的模型
model = joblib.load("rsf_all.pkl")

#加载通过训练数据集得到的特征解释
filename_expl = 'explainer_all.sav'
explainer = pickle.load(open(filename_expl, 'rb'))

#设置界面
st.title("CVD patient heart failure survival and hazard risk prediction")
# st.write('本应用可根据患者的基本数据预测未来生存概率和发病风险')
st.sidebar.title("Patient Data")
age = st.sidebar.slider(label="年龄(Age)",min_value=30, max_value=90,step=1)
sex =st.sidebar.radio("性别(Gender)",options=['男(male)','女(Female)'],index=0)
anaemia=st.sidebar.checkbox("贫血(Anaemia)", 0)
hbp = st.sidebar.checkbox("高血压(High Blood Pressure)",0)
smoking = st.sidebar.checkbox("吸烟(Smoking)",0)
diabets = st.sidebar.checkbox("糖尿病(Diabetes)",0)
cpk = st.sidebar.slider("肌酐磷酸激酶CPK(mcg/L)",min_value=10, max_value=10000, step=1)

ejection_fraction=st.sidebar.slider("射血分数(Ejection Fraction)(%)",min_value=10,max_value=100)
platelets = st.sidebar.slider("血小板(kiloplatelets/mL)",min_value=20000,max_value=90000,step=100)
serum_creatinine = st.sidebar.slider("血清肌酐(Serum Creatinine)(mg/mL)",min_value=0.5,max_value=10.0,step=0.1)
serum_sodium = st.sidebar.slider("血清钠(Serum Sodium)(mEq/L)",min_value=100,max_value=150,step=1)
features = {'age': age,
            'anaemia': anaemia,
            'creatinine_phosphokinase':cpk,
            'diabetes': diabets,
            'ejection fraction':ejection_fraction,
            'high_blood_pressure':hbp,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium,
            'sex': sex,
            'smoking':smoking
        }

features_cn = {'Age': age, 'Gender': sex,
        'Anaemia': anaemia, 'High Blood Pressure': hbp,
        'Diabetes': diabets, 'Smoking': smoking,
        'Ejection Fraction(%)': ejection_fraction, 'CPK(mcg/L)': cpk,
        'Platelets(kiloplatelets/mL)': platelets, 'Serum Creatinine(mg/mL)': serum_creatinine, 'Serum Sodium(mEq/L)': serum_sodium
        }

fig, ax= plt.subplots()
features_df  = pd.DataFrame([features])
features_df['anaemia']=features_df['anaemia'].apply(lambda x:1 if x==True else 0)
features_df['high_blood_pressure']=features_df['high_blood_pressure'].apply(lambda x:1 if x==True else 0)
features_df['diabetes']=features_df['diabetes'].apply(lambda x:1 if x==True else 0)
features_df['smoking']=features_df['smoking'].apply(lambda x:1 if x==True else 0)
features_df['sex']=features_df['sex'].apply(lambda x:1 if x=='男(male)' else 0)


df_cn = pd.DataFrame([features_cn])
st.write("Current Patient Data")
st.dataframe(df_cn,width=1200)


if st.button('Predict'):
    with st.spinner(text='Calculating......'):
        surv = predict_survival(model, features_df)
        hazard = predict_hazard(model,features_df)
        shaps = explainer(features_df)

        df_surv=pd.DataFrame(data=surv,columns=model.event_times_)
        df_hazard=pd.DataFrame(data=hazard,columns=model.event_times_)

        
        df = pd.concat([df_surv,df_hazard])

        
        df = df.T
        df.columns=['Survival Probability','Cumulative Hazard Risk']
        fig = px.line(df)
        fig.update(layout=dict(xaxis=dict(title="Days", tickangle=-30,
                           showline=True, nticks=20),
                           yaxis=dict(title="Probability", showline=True)))
        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_layout(legend_title_text='Class',hovermode='x unified')

        col1, col2 = st.columns((1,1))
        with col1:
            st.write("Feature Impact")
            st.pyplot(shap.plots.waterfall(shaps[0]))

        with col2:
            st.write("Survival Probability and Cumulative Hazard Risk")
            st.plotly_chart(fig)
