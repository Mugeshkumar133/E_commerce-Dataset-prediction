import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
from streamlit_option_menu import option_menu
import re
import plotly.express as px
st.set_page_config(page_title ="E-commerce Dataset Prediction",layout="wide")

st.write("""

<div style='text-align:center'>
    <h1 style='color:#009999;'>E-commerce Classification Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
     opt = option_menu("E-commerce Data",
                      ["Algorithms","Prediction"],
                      menu_icon="cast",
                      styles={
                          "container":{"padding":"4!important","background-color":"gray"},
                          "icon":{"color":"red","font-size":"20px"},
                          "nav-link":{"font-size":"20px","text-align":"left"},
                          #"nav-link-selected":{"background-color":"yellow"}
                      })



if opt == "Algorithms":
    df = pd.DataFrame({
        "Algorithm Names":["Decision Tree","Logistic Regression ","Random Forest"],
        "Accuracy":[94,78,96],
        "Precision":[94,82,95],
        "Recall":[93,60,95],
        "F1_score":[93,69,95]
    })

    st.table(df)

if opt == "Prediction":
 with st.form("my_form"):
    c1,c2 = st.columns(2)


    st.markdown(
        """
        <style>
            .st-ax{
                background-color: lightblue;
            }

            .stTextInput input{
                background-color: lightblue;
            }
            .stNumberInput input{
                background-color: lightblue;
            }

            .stDateInput Input{
                background-color: lightblue;
            }

        </style>
        """
    ,unsafe_allow_html=True
    )

    with c1:
        st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
        transactionRevenue = st.text_input("Enter transactionRevenue (Min:0.0 & Max:307221222.5)")
        num_interactions = st.text_input("Enter num_interactions (Min:20.0 & Max:25911.0)")
        count_hit = st.text_input("Enter count_hit (Min:2, Max:7085.0)")
        historic_session_page = st.text_input("historic_session_page (Min:0.0, Max:5021.25)")
        time_on_site = st.text_input("time_on_site (Min:0.0, Max:26652.75)")

    with c2:
        avg_session_time = st.text_input("avg_session_time (Min:2.0, Max:1109.1536494755242)")
        avg_session_time_page = st.text_input("avg_session_time_page (Min:0.0, Max:339.53914141414145)")
        historic_session = st.text_input("historic_session (Min:2.0, Max:23271.5)")
        visits_per_day = st.text_input("visits_per_day (Min:0.9230769230769232, Max:304.8440708626405)")
        submit_button = st.form_submit_button(label="PREDICT STATUS")
        st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    # flag=0
    # pattern = "^(?:\d+|\d*\.\d+)$"
    for i in ["transactionRevenue","num_interactions","count_hit","historic_session_page","time_on_site","avg_session_time","avg_session_time_page","historic_session","visits_per_day"]:
    #     if re.match(pattern, i):
    #         pass
    #     else:
    #         flag=1
    #         break
        # if submit_button :
        #     if len(i)==0:
        #       st.write("please enter a valid number and space  is not allowed")
        #     else:
        #       st.write("You have entered an invalid value: ",i)

        if submit_button :
            import pickle
            with open(r"C:/Users/muges/Downloads/cmodel.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r"C:/Users/muges/Downloads/cscaler.pkl", 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"C:/Users/muges/Downloads/ct.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

        # Predict the has_converted for a new sample
        # transactionRevenue,num_interactions,count_hit,historic_session_page,time_on_site,avg_session_time,avg_session_time_page,historic_session,visits_per_day
            new_sample = np.array([[float(transactionRevenue),float(num_interactions),float(count_hit),float(historic_session_page),float(time_on_site),float(avg_session_time),float(avg_session_time_page),float(historic_session),float(visits_per_day)]])
            # new_sample_ohe = t_loaded.transform(new_sample[:, [9]]).toarray()
            new_sample = np.array((new_sample[:, [0,1,2, 3, 4, 5, 6,7,8]]))
            new_sample = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample)
            if new_pred== 1:
                 st.write('## :green[The Status is Converted] ')
                 break
            else:
                 st.write('## :red[The Status is Not Converted] ')
                 break

uploaded_file=st.sidebar.file_uploader(label="Upload your csv or excel file.max(200mb)",type=["csv","xlsx"])

if uploaded_file is not None:
    print(uploaded_file)
    
    try:
       df=pd.read_csv(uploaded_file)
    except Exception as e:
       print(e)
       df=pd.read_excel(uploaded_file)
try:
    st.write(df)
    numeric_columns= list(df.select_dtypes(["float","int"]).columns)
    categorical_column=list(df.select_dtypes("object").columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")
#add a select widget tot the  sidebar
chart_select=st.sidebar.selectbox(label="select the chart type",
                                  options=["Scatterplots","Barcharts","Boxplot","Histogram"])

if chart_select=="Scatterplots":
    st.sidebar.subheader("Scatterplot settings")
    try:
        x_values= st.sidebar.selectbox("X axis",options=numeric_columns)
        y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
        plot=px.scatter(data_frame=df,x=x_values,y=y_values)
        #display the chart 
        st.plotly_chart(plot)
    except Exception as e:
        print(e)


if chart_select=="Barcharts":
    st.sidebar.subheader("Barcharts settings")
    try:
        x_values= st.sidebar.selectbox("X axis",categorical_column)
        y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
        plot=px.bar(data_frame=df,x=x_values,y=y_values)
        #display the chart 
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select=="Boxplot":
    st.sidebar.subheader("Boxplot settings")
    try:
        x_values= st.sidebar.selectbox("X axis",categorical_column)
        y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
        plot=px.box(data_frame=df,x=x_values,y=y_values)
        #display the chart 
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select=="Histogram":
    st.sidebar.subheader("Barcharts settings")
    try:
        # x_values= st.sidebar.selectbox("X axis",categorical_column)
        y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
        plot=px.histogram(data_frame=df,x=y_values,nbins=20)
        #display the chart 
        st.plotly_chart(plot)
    except Exception as e:
        print(e)



st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Mugeshkumar</h6>', unsafe_allow_html=True )