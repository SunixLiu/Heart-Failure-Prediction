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
# import joblib
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
   page_title="CVD patient heartfailure survival and hazard prediction",
   page_icon="ð§",
   layout="wide",
   initial_sidebar_state="expanded",
)

#é¢æµçå­æ¦ç
def predict_survival(model, df):
    predicted_data= model.predict_survival_function(df,return_array=True)
    return predicted_data

#é¢æµç´¯ç§¯åçé£é©
def predict_hazard(mode,df):
    predicted_data = model.predict_cumulative_hazard_function(df,return_array=True)
    return predicted_data

#å è½½è®­ç»å¥½çæ¨¡å
# model = joblib.load("rsf_all.jl")
model = pickle.load(open('rsf_all.pkl','rb'))

#å è½½éè¿è®­ç»æ°æ®éå¾å°çç¹å¾è§£é
# filename_expl = 'explainer_all.jl'
explainer = pickle.load(open('explainer_all.sav', 'rb'))

#è®¾ç½®çé¢
st.title("CVD patient heart failure survival and hazard risk prediction")
# st.write('æ¬åºç¨å¯æ ¹æ®æ£èçåºæ¬æ°æ®é¢æµæªæ¥çå­æ¦çååçé£é©')
st.sidebar.title("Patient Data")
age = st.sidebar.slider(label="å¹´é¾(Age)",min_value=30, max_value=90,step=1)
sex =st.sidebar.radio("æ§å«(Gender)",options=['ç·(male)','å¥³(Female)'],index=0)
anaemia=st.sidebar.checkbox("è´«è¡(Anaemia)", 0)
hbp = st.sidebar.checkbox("é«è¡å(High Blood Pressure)",0)
smoking = st.sidebar.checkbox("å¸ç(Smoking)",0)
diabets = st.sidebar.checkbox("ç³å°¿ç(Diabetes)",0)
cpk = st.sidebar.slider("èéç£·é¸æ¿é¶CPK(mcg/L)",min_value=10, max_value=10000, step=1)

ejection_fraction=st.sidebar.slider("å°è¡åæ°(Ejection Fraction)(%)",min_value=10,max_value=100)
platelets = st.sidebar.slider("è¡å°æ¿(kiloplatelets/mL)",min_value=20000,max_value=90000,step=100)
serum_creatinine = st.sidebar.slider("è¡æ¸èé(Serum Creatinine)(mg/mL)",min_value=0.5,max_value=10.0,step=0.1)
serum_sodium = st.sidebar.slider("è¡æ¸é (Serum Sodium)(mEq/L)",min_value=100,max_value=150,step=1)
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
features_df['sex']=features_df['sex'].apply(lambda x:1 if x=='ç·(male)' else 0)


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
