import streamlit as st

import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()
#Load the saved model
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

clf_loaded = data["model"]
st.set_page_config(page_title="cancer Prediction",layout="centered",initial_sidebar_state="expanded")
def preprocess(radius_mean,perimeter_mean,area_mean,symmetry_mean,compactness_mean,concave_points_mean  ):   
 
 
 
   
    user_input=[radius_mean,perimeter_mean,area_mean,symmetry_mean,compactness_mean,concave_points_mean ]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scal.fit_transform(user_input)
    prediction = clf_loaded.predict(user_input)

    return prediction
       
# front end elements of the web pperimeter_mean 
html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Breast Cancer Prediction </h1> 
    </div> 
    """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 

      

# following lines create boxes in which user can enter data required to make prediction
radius_mean=st.number_input('radius_mean')
perimeter_mean=st.number_input('perimeter_mean')
area_mean=st.number_input('area_mean')
symmetry_mean=st.number_input('symmetry_mean')
compactness_mean=st.number_input('compactness_mean')
concave_points_mean=st.number_input('concave_points_mean')



pred=preprocess(radius_mean,perimeter_mean,area_mean,symmetry_mean,compactness_mean,concave_points_mean )



if st.button("Predict"):    
  if pred[0] == 0:
    st.error('Warning! You have Breast Cancer!')
    
  else:
    st.success('You have lower risk to have Breast Cancer!')