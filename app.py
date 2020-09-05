#The program will use machine learning with python gathering users ,edical levels to predict if the patient has diabetes

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd


#will be using streamlit to produce visual input and results for the user 
st.write("""
# Diabetes Detection
Using Patient Data and ML to predict Diabates   
""")

#setting the main image and siplaying it
img = Image.open('Main_image.png')
st.image(img,use_column_width=True)



#get the test data csv to use to train the model. Found on kaggle
data_frame = pd.read_csv('dataset.csv')


#create header for data
st.subheader('DataSet Information')


#display data table
st.dataframe(data_frame)


#display stats from the datafram for overview
st.write(data_frame.describe())


#show data chart
chart = st.bar_chart(data_frame)




#get data into independent x and y values 
x = data_frame.iloc[:, 0:8].values
y = data_frame.iloc[:, -1].values

#make the dataset 75 percent for training and 25 percent for the testing component
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#create input to receive data from user
def get_input():
    #creating input sliders
    pregnancies = st.sidebar.slider('Pregnancies', 0,17,3)
    glucose = st.sidebar.slider('Glucose', 0,199,120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0,125,75)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0,99,23)
    insulin = st.sidebar.slider('Insulin', 0.0,1000.0,100.0)
    bmi = st.sidebar.slider('BMI', 0.0,70.0,30.0)
    dpf = st.sidebar.slider('DPF', 0.0,2.50,0.3)
    age = st.sidebar.slider('Age', 0,100,20)

    #create a dictionary variable 
    patient_data = {
        'pregnancies':pregnancies,
        'glucose':glucose,
        'blood_pressure':blood_pressure,
        'skin_thickness':skin_thickness,
        'insulin':insulin,
        'bmi':bmi,
        'dpf':dpf,
        'age':age
    }

    #turn patient_data into a dataframe and return it
    feature = pd.DataFrame(patient_data,index=[0])
    return feature




#store patient input
patient_input = get_input()
#set subheader for patient_input
st.subheader('Patient Input')
#display patient input
st.write(patient_input)


#create and train the randomforestclassifier model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)


#display the final models metrics
st.subheader('Model Accuracy Score')
st.write(str(accuracy_score(y_test,RandomForestClassifier.predict(x_test))*100)+'%')


#store models prediction to use for ouput
predict = RandomForestClassifier.predict(patient_input)


#display prediction
st.subheader('Diabates Classification: ')
st.write(predict)





