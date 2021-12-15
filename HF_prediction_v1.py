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
   page_title="心衰患者生存概率和发病风险预测演示",
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
st.title("心衰患者生存概率和发病风险预测")
st.write('本应用可根据患者的基本数据预测未来生存概率和发病风险')
st.sidebar.title("患者数据")
age = st.sidebar.slider(label="年龄",min_value=30, max_value=90,step=1)
sex =st.sidebar.radio("性别",options=['男','女'],index=0)
anaemia=st.sidebar.checkbox("贫血", 0)
hbp = st.sidebar.checkbox("高血压",0)
smoking = st.sidebar.checkbox("吸烟",0)
diabets = st.sidebar.checkbox("糖尿病",0)
cpk = st.sidebar.slider("肌酐磷酸激酶CPK(mcg/L)",min_value=10, max_value=10000, step=1)

ejection_fraction=st.sidebar.slider("射血分数(%)",min_value=10,max_value=100)
platelets = st.sidebar.slider("血小板(kiloplatelets/mL)",min_value=20000,max_value=90000,step=100)
serum_creatinine = st.sidebar.slider("血清肌酐(mg/mL)",min_value=0.5,max_value=10.0,step=0.1)
serum_sodium = st.sidebar.slider("血清钠(mEq/L)",min_value=100,max_value=150,step=1)
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

features_cn = {'年龄': age, '性别': sex,
        '贫血症': anaemia, '高血压': hbp,
        '糖尿病': diabets, '吸烟': smoking,
        '射血分数(%)': ejection_fraction, '肌酐磷酸激酶(mcg/L)': cpk,
        '血小板(kiloplatelets/mL)': platelets, '血清肌酐(mg/mL)': serum_creatinine, '血清钠(mEq/L)': serum_sodium
        }

fig, ax= plt.subplots()
features_df  = pd.DataFrame([features])
features_df['anaemia']=features_df['anaemia'].apply(lambda x:1 if x==True else 0)
features_df['high_blood_pressure']=features_df['high_blood_pressure'].apply(lambda x:1 if x==True else 0)
features_df['diabetes']=features_df['diabetes'].apply(lambda x:1 if x==True else 0)
features_df['smoking']=features_df['smoking'].apply(lambda x:1 if x==True else 0)
features_df['sex']=features_df['sex'].apply(lambda x:1 if x=='男' else 0)

#显示选择好的患者数据
df_cn = pd.DataFrame([features_cn])
st.write("患者数据")
st.dataframe(df_cn,width=1200)

#点击按钮后：
if st.button('预测'):
    with st.spinner(text='计算中，请稍等......'):
        surv = predict_survival(model, features_df)
        hazard = predict_hazard(model,features_df)
        shaps = explainer(features_df)

        df_surv=pd.DataFrame(data=surv,columns=model.event_times_)
        df_hazard=pd.DataFrame(data=hazard,columns=model.event_times_)

        #把生存概率和累积风险数据合并到一个dataframe
        df = pd.concat([df_surv,df_hazard])

        #做一个矩阵旋转
        df = df.T
        df.columns=['生存概率','累计风险']
        fig = px.line(df)
        fig.update(layout=dict(xaxis=dict(title="天数", tickangle=-30,
                           showline=True, nticks=20),
                           yaxis=dict(title="概率", showline=True)))
        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_layout(legend_title_text='类别',hovermode='x unified')

        col1, col2 = st.columns((1,1))
        with col1:
            st.write("影响特征解释")
            st.pyplot(shap.plots.waterfall(shaps[0]))

        with col2:
            st.write("心衰患者生存概率与累积风险预测")
            st.plotly_chart(fig)
