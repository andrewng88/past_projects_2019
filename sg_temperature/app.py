import streamlit as st
import pandas as pd
import numpy as np
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

st.title('Interactive SG Total Rain Fall ( monthly ) Prediction App')
st.subheader('Model and Deployed by Andrew Ng')

DATA_URL =('./data/fb_ts.csv')



month= st.slider('Month of prediction:',1,12)

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    #data = pd.read_csv(DATA_URL,parse_dates=['ds'])
    return data

data = load_data()

def plot_fig():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name="Rain_Fall_mm"))
	fig.layout.update(title_text='Time Series for SG Total Rain Fall - model on the fly',xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	return fig

if st.checkbox('Show Rain Fall in table format'):
    st.subheader('Raw data')
    st.write(data)
	
# plotting the figure of Actual Data
plot_fig()

# code for facebook prophet prediction

m = Prophet(yearly_seasonality=True,seasonality_mode='multiplicative')
m.fit(data)
future = m.make_future_dataframe(periods=month,freq ='MS')
forecast = m.predict(future)

#plot forecast
fig1 = plot_plotly(m, forecast)
if st.checkbox('Show forecast data'):
     st.subheader('forecast data')
     st.write(forecast.tail(12))
text = 'Forecasting Rain Fall in SG for {month} month in the year 2020'
st.write(text)
st.plotly_chart(fig1)

#plot component wise forecast
st.write("Component wise forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)