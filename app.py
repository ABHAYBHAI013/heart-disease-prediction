import streamlit as st 
import pandas as pd 
import joblib 

model = joblib.load('LogisticRegression_heart.pkl')
scaler = joblib.load('Scaler.pkl')
expected_columns = joblib.load('columns.pkl') 



st.title('Heart stroke prediction by ABHAY SOLANKI ❤️❤️') 
st.markdown('provide the following details') 
age = st.slider('age',18,100,40)
sex = st.selectbox('sex',['male','female'])
chest_pain = st.selectbox('chest pain type',['ATA',"NAP","TA","ASY"]) 
Resting_bp = st.number_input('Resting blood pressure(mm hg)', 80,200,120) 
cholestrol = st.number_input('cholestrol (mg / dl)',100,600,200)
fasting_bs = st.selectbox('fasting blood sugar > 120 mg/dl',[0,1]) 
resting_ecg = st.selectbox('Resting ecg ',['Normal','ST','LVH']) 
max_hr = st.slider('max heart rate', 60, 220, 150 ) 
exercise_angina = st.selectbox('exercise-Induced Angina',['Y','N']) 
oldpeak = st.slider('oldpeak(ST depression)',0.0,6.0,1.0) 
st_slope = st.selectbox('st slope',['UP','Flat','Down']) 


if st.button('Predict'):
    raw_input = {
    'Age': age,
    'RestingBP': Resting_bp,
    'Cholesterol': cholestrol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_' + sex: 1,
    'ChestPainType_' + chest_pain: 1,
    'RestingECG_' + resting_ecg: 1,
    'ExerciseAngina_' + exercise_angina: 1,
    'ST_Slope_' + st_slope: 1
}
    

    input_df = pd.DataFrame([raw_input]) 

    for col in expected_columns: 
        if col not in input_df.columns:
            input_df[col] = 0 
    
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df) 
    prediction = model.predict(scaled_input)[0]   



    if prediction == 1:
        st.error('💀 HIGH RISK OF HEART DISEASE ') 
    else:
        st.success('😊😊 LOW RISK OF HEART DISEASE') 
