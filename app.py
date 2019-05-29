# import dependencies
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
import plotly.graph_objs as go
from flask import Flask, render_template, request, json, jsonify

app = Flask(__name__)

# Create your connections with indoor and outdoor databases
Ocnx = sqlite3.connect('Outdoor.db', check_same_thread=False)
Icnx = sqlite3.connect('Indoor.db', check_same_thread=False)

def getData():

    curs=Ocnx.cursor()
    for row in curs.execute("SELECT * FROM BME_DATA ORDER BY TIME_STAMP DESC LIMIT 1"):
        Temperature = row[2]
    Ocnx.close()
    return Temperature


@app.route("/")
def index():
    Temperature = getData()
    templateData ={'Temperature': Temperature}
    return render_template('index.html', **templateData)


@app.route("/analysis")
def analysis():
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
    ind_df = ind_forecast.to_dict(orient='records')
    return jsonify(ind_df)

if __name__ == '__main__':
    app.run(debug=True)