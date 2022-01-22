import streamlit as st
import joblib
import numpy as np

st.markdown(
      """
   	<style>
     .main {
     background-color: #f4e1e6;
    text-align: center;
    font-size: 16px;
     }

     .reportview-container .block-container{
        max-width: 800px;
        padding-top: 1rem;
        padding-bottom: 3rem;

    }
    .navbar {
    background-color: #333333;
    font-size: large;
    color: white
	}

	.result {
    font-size: 20px;
    font-weight: bold;
	}
	footer {
	
	visibility: hidden;
	
	}
	footer:after {
		content:'Copyright © 2021'; 
		visibility: visible;
		display: block;
		position: relative;
		padding: 5px;
		top: 2px;
	}

	</style>
      """,
      unsafe_allow_html=True
  )

siteHeader = st.beta_container()
footer = st.beta_container()

with siteHeader:
	st.markdown('<p class="navbar">Diabetes Prediction</p>', unsafe_allow_html=True)
	st.text('Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. ')

	with st.form(key='columns_in_form'):
		input_col1, input_col2 = st.beta_columns(2)

		preg= input_col1.number_input('Pregnancies', 
		  min_value=0, 
		  max_value=10, 
		  value=0, 
		  step=1)
		gluc= input_col1.number_input('Glucose', min_value=0, max_value=200,value=0)
		bp= input_col1.number_input('Blood Pressure', min_value=0, max_value=125, value=0)
		skt= input_col1.number_input('Skin Tickness', min_value=0, max_value=100, value=0)

		ins= input_col2.number_input('Insulin', min_value=0, max_value=800, value=0)
		bmi= input_col2.number_input('BMI', min_value=0, max_value=70, value=0)
		dpf= input_col2.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, format="%.2f")
		age= input_col2.number_input('Age', value=0)

		submitted = st.form_submit_button('Submit')


	model = joblib.load("model")

	@st.cache
	def ValuePredictor(to_predict_list):
	    to_predict = np.array(to_predict_list).reshape(1,len(to_predict_list))
	    result = model.predict(to_predict)
	    return result[0]

	if submitted:
		to_predict_list=[preg,gluc,bp,skt,ins,bmi,dpf,age]
		to_predict_list = list(map(float, to_predict_list))
		result = ValuePredictor(to_predict_list)
		if(int(result)==1):
			st.markdown('<p class="result">Sorry ! Suffering</p>', unsafe_allow_html=True)
		else:
			st.markdown('<p class="result">Congrats ! you are Healthy</p>', unsafe_allow_html=True)