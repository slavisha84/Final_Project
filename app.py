# import dependencies
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
import plotly.graph_objs as go
from flask import Flask, render_template, request, json, jsonify
import plotly.io as pio

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io

app = Flask(__name__)

# Create your connections with indoor and outdoor databases
Ocnx = sqlite3.connect('Outdoor.db', check_same_thread=False)
Icnx = sqlite3.connect('Indoor.db', check_same_thread=False)

def get_ind_Data():
# Get inside data and produce C and F values 
    ind_curs=Icnx.cursor()
    for row in ind_curs.execute("SELECT * FROM BME_DATA ORDER BY TIME_STAMP DESC LIMIT 1"):
        Ind_TemperatureC = row[2]
        Ind_TemperatureF = round((row[2]* 9/5) + 32)
        Ind_TemperatureC2 = row[2]+6
        Ind_TemperatureF2 = round((row[2]* 9/5) + 32)+6
        Ind_TemperatureC3 = row[2]+4
        Ind_TemperatureF3 = round((row[2]* 9/5) + 32)+4
        Ind_TemperatureC4 = row[2]+8
        Ind_TemperatureF4 = round((row[2]* 9/5) + 32)+8

    #Icnx.close()
    return Ind_TemperatureC, Ind_TemperatureF, Ind_TemperatureC2, Ind_TemperatureF2, Ind_TemperatureC3, Ind_TemperatureF3, Ind_TemperatureC4, Ind_TemperatureF4

def get_out_Data():
# Get inside data and produce C and F values 
    out_curs=Ocnx.cursor()
    for row in out_curs.execute("SELECT * FROM BME_DATA ORDER BY TIME_STAMP DESC LIMIT 1"):
        Out_TemperatureC = row[2]
        Out_TemperatureF = round((row[2]* 9/5) + 32)
    #Ocnx.close()
    return Out_TemperatureC, Out_TemperatureF


@app.route("/")
def index():
    Ind_TemperatureC, Ind_TemperatureF, Ind_TemperatureC2, Ind_TemperatureF2,Ind_TemperatureC3, Ind_TemperatureF3,Ind_TemperatureC4, Ind_TemperatureF4 = get_ind_Data()
    Out_TemperatureC, Out_TemperatureF = get_out_Data()
    templateData = {'Ind_TemperatureC': Ind_TemperatureC,
                    'Ind_TemperatureF': Ind_TemperatureF,
                    'Out_TemperatureC': Out_TemperatureC,
                    'Out_TemperatureF': Out_TemperatureF,
                    'Ind_TemperatureC2': Ind_TemperatureC2,
                    'Ind_TemperatureF2': Ind_TemperatureF2,
                    'Ind_TemperatureC3': Ind_TemperatureC3,
                    'Ind_TemperatureF3': Ind_TemperatureF3,
                    'Ind_TemperatureC4': Ind_TemperatureC4,
                    'Ind_TemperatureF4': Ind_TemperatureF4}
    return render_template('index.html', **templateData)


@app.route("/analytics" , methods=['POST'])
def plot_temp():
    Outdoor_df = pd.read_sql_query("SELECT * FROM BME_DATA", Ocnx)
    Indoor_df = pd.read_sql_query("SELECT * FROM BME_DATA", Icnx)

    # Cleaning up Outdoor Data
    Outdoor_df['TIME_STAMP'] = pd.to_datetime(Outdoor_df['TIME_STAMP'])
    Outdoor_df['TIME_STAMP'] = Outdoor_df['TIME_STAMP'].dt.round('60min')
    Outdoor_df = Outdoor_df.groupby(['TIME_STAMP'], as_index=False)["TEMPERATURE","GAS","HUMIDITY", "PRESSURE", "ALTITUDE"].mean()
    Outdoor_df = Outdoor_df.loc[:,["TIME_STAMP", "TEMPERATURE", "GAS", "HUMIDITY", "PRESSURE"]]
    Outdoor_df["GAS"] = round(Outdoor_df["GAS"],1)
    Outdoor_df["HUMIDITY"] = round(Outdoor_df["HUMIDITY"],1)
    Outdoor_df["TEMPERATURE"] = round(Outdoor_df["TEMPERATURE"],1)
    Outdoor_df["PRESSURE"] = round(Outdoor_df["PRESSURE"],1)
    Outdoor_df['TEMPERATURE'] = round((Outdoor_df['TEMPERATURE']* 9/5) + 32)

    # Cleaning up Indoor Data
    Indoor_df['TIME_STAMP'] = pd.to_datetime(Indoor_df['TIME_STAMP'])
    Indoor_df['TIME_STAMP'] = Indoor_df['TIME_STAMP'].dt.round('60min')
    Indoor_df = Indoor_df.groupby(['TIME_STAMP'], as_index=False)["TEMPERATURE","GAS","HUMIDITY", "PRESSURE", "ALTITUDE"].mean()
    Indoor_df = Indoor_df.loc[:,["TIME_STAMP", "TEMPERATURE", "GAS", "HUMIDITY", "PRESSURE"]]
    Indoor_df["GAS"] = round(Indoor_df["GAS"],1)
    Indoor_df["HUMIDITY"] = round(Outdoor_df["HUMIDITY"],1)
    Indoor_df["TEMPERATURE"] = round(Indoor_df["TEMPERATURE"],1)
    Indoor_df["PRESSURE"] = round(Indoor_df["PRESSURE"],1)
    Indoor_df['TEMPERATURE'] = round((Indoor_df['TEMPERATURE']* 9/5) + 32)

    # Merging Indor and Outdoor (on time stamp)
    master_df = pd.merge(Outdoor_df, Indoor_df, on = "TIME_STAMP", how = "left", suffixes=("_Out","_In"))
    master_df = master_df.dropna()

    # Creating indoor and Outdoor Datasets
    outdoor = master_df.loc[:,["TIME_STAMP", "TEMPERATURE_Out"]]
    indoor = master_df.loc[:,["TIME_STAMP", "TEMPERATURE_In"]]
    outdoor.columns = ['ds', 'y']
    indoor.columns = ['ds', 'y']

    # Building the model for inodor and outdoor
    od = Prophet()
    od.fit(outdoor)
    ind = Prophet()
    ind.fit(indoor)

    # Outdoor forecasting
    od_future = od.make_future_dataframe(periods= 1)
    od_forecast = od.predict(od_future)

    # Indoor forecasting
    ind_future = ind.make_future_dataframe(periods= 1)
    ind_forecast = ind.predict(ind_future)
    #ind_df = ind_forecast.to_dict(orient='records')
    #return jsonify(ind_df)
    od_plot = od.plot(od_forecast)
    od_plot.savefig('static/od_plot.png')

    ind_plot = ind.plot(ind_forecast)
    ind_plot.savefig('static/ind_plot.png')

    od_fig1 = od.plot_components(od_forecast)
    od_fig1.savefig('static/od_fig1.png')

    ind_fig1 = ind.plot_components(ind_forecast)
    ind_fig1.savefig('static/ind_fig1.png')   

# Producing full plot for outdoor data
    fig5 = go.Figure([
        go.Scatter(x=outdoor['ds'], y=outdoor['y'], name='y'),
        go.Scatter(x=od_forecast['ds'], y=od_forecast['yhat'], name='yhat'),
        go.Scatter(x=od_forecast['ds'], y=od_forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
        go.Scatter(x=od_forecast['ds'], y=od_forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
        go.Scatter(x=od_forecast['ds'], y=od_forecast['trend'], name='Trend')
        ])
    pio.write_image(fig5, "static/Outdoor_f.png", width=800, height=600, scale=4)

# Producing full plot for indoor data
    fig6 = go.Figure([
        go.Scatter(x=indoor['ds'], y=indoor['y'], name='y'),
        go.Scatter(x=ind_forecast['ds'], y=ind_forecast['yhat'], name='yhat'),
        go.Scatter(x=ind_forecast['ds'], y=ind_forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
        go.Scatter(x=ind_forecast['ds'], y=ind_forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
        go.Scatter(x=ind_forecast['ds'], y=ind_forecast['trend'], name='Trend')
        ])
    pio.write_image(fig6, "static/Indoor_f.png", width=800, height=600, scale=4)

    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True)     