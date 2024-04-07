import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


#reading the data on LAT and LONG of all the MRT stations in singapore

data=pd.read_csv('mrt.csv')
mrt_location=pd.DataFrame(data)

#this is the configuration page for streamlit application
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    #page_icon=""
    layout="wide"
)

#creating the sidebar in streamlit application
with st.sidebar:
    selected=option_menu("Main Menu",["About Project","Predictions"],
                         icons=["house","gear"],
                         styles={"nav-link":{"font":"sans serif","font_size":"20px","text-align":"centre"},
                                 "nav-link-selected":{"font":"sans serif","background-color":"#0072b1"},
                                 "icon":{"font-size":"20px"}
                                 }
                        ) 
    

#project section

    if selected =="About Project":
        st.markdown("#:blue[Singapore Resale Flat Prices Prediction]")
        st.markdown('<div style="height: 50px;"></div>',unsafe_allow_html=True)
        st.markdown("###:blue[Technologies:] python,pandas,Numpy,Scikit-learn,streamlit,python scripting,"
                    "Machine Learning,Data Preprocessing ,Visualization,EDA,Model Building,Data Wrangling,"
                    "Model Deployment")
        
        st.markdown("### :blue[Overview:] This project aims to construct the machine learning model and implement"
                    "it as a user-friendly online application in order to provide accurate predictions about the"
                    "resale values of the apartments in singapore .This prediction model will be based on the past transactions"
                    "involving resale flats , and its goal is to aid both future buyers and sellers evealuating the"
                    "worth of a flat after it has been previously resold.Reasale prices are influenced by a wide variety"
                    "of criteria , including location , the kind of apartment , the total square footage , and the length "
                    "of the lease .The provision of customers with an expected resale price based on the criteria is "
                    "on the ways  in which a prdictive model may assist in the overcoming of the obstacles.")
        
        st.markdown("###:blue[Domain:]Real Estate")


#predictions section
if selected =="Predictions":
    st.markdown("# :blue[Predicting the Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form ("form1"):
            
            #new data inputs from the user for preicting the resale price 
            street_name=st.text_input("Street Name")
            block=st.text_input("Block Number")
            floor_area_sqm=st.number_input('Floor Area (Per Square Meter)',min_value=1.0,max_value=500.0)
            lease_commence_date=st.number_input('Lease Commence Date ')
            storey_range=st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

#submit Button for predict resale price 
        
        submit_button=st.form_submit_button(label-"PREDICT RESALE PRICE")

        if submit_button is not None:
            with open(r"RandomForestRegressor_model.pkl",'rb') as file:
                loaded_model=pickle.load(file)
            #with open(r".pkl",'rb') as f:
                #scalar_loaded=pickle.load(f)

#calculating lease remaining_years using lease_commence date 
            
            lease_remain_years=99-(2023-lease_commence_date)

#calculating median of storey_range to make our calculation quite comfirtable 

            split_list=storey_range.split(' TO ')
            float_list=[float(i) for i in split_list]
            storey_median=statistics.median(float_list)

#getting the address by joining the block number and the street name 
            
            address=block +" "+ street_name
            query_address=address 
            query_string='https://developers.onemap.sg/commonapi/search?searchVal='+str(query_address)+'&returnGeom=Y&getAddressDetails=Y'
            resp=requests.get(query_string)

#using openmap API getting the latitude and longitutde location of that address 

            origin=[]
            data_geo_location=json.loads(resp.content)
            if data_geo_location['found']!=0:
                latitude=data_geo_location['results'][0]['LATITUDE']
                longitude=data_geo_location['results'][0]['LONGITUDE']
                origin.append(latitude,longitude)

#Appending the Latitudes and longitudes of the MRT stations 
# Latitde and longitude are been append inthe form of tuble to the list 

            mrt_lat=mrt_location['latitude']
            mrt_long=mrt_location['longitude'] 
            list_of_mrt_coordinates.append((lat,long))


#Getting distance to the nearest MRT stations 

            list_of_dist_mrt=[]
            for destination in range(0, len(list_of_mrt_coordinates[destination]).meters):
                shortest=(min(list_of_dist_mrt)) 
                min_dist_mrt=shortest
                list_of_dist_mrt.clear()

#getting distance from CBD

            cbd_dist=geodesic(origin,(1.2830,103.8513)).meters


#sending the user enter values for prediction to our model 

            new_sample=np.array(
            [[cbd_dist,min_dist_mrt,np.log(floor_area_sqm),lease_reamin_years,np.log(storey_median)]])
            #new_sample=scaler_loaded.transform(new_sample[:, :5])
            new_pred=loaded_model.predict(new_sample)[0]
            st.write('## : green[Predicted resale price:]',np.exp(new_pred))     


except Exception as e:
    st.write("Enter the above values to get the predicted resale price of the flat")                  
