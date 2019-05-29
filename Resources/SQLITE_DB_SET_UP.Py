# Import Dependencies
import sqlite3 as lite
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String, Float, DATETIME, NUMERIC
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import *
import datetime

# Create engine and automap_base
engine = create_engine('sqlite:///Sensors.db', echo=True)
conn = engine.connect()

# Create Table BME_DATA

Base = declarative_base()

class Sensors(Base):
    __tablename__ = 'BME_DATA'
    
    id = Column(Integer, primary_key=True)
    TIME_STAMP = Column(DATETIME)
    TEMPERATURE = Column(NUMERIC)
    GAS = Column(NUMERIC)
    HUMIDITY = Column(NUMERIC)
    PRESSURE = Column(NUMERIC)
    ALTITUDE = Column(NUMERIC)
    
Base.metadata.create_all(conn)

# Creating dummy data set for test
# ----------------------------------
now = datetime.datetime.now()
test = Sensors(TIME_STAMP =(now), TEMPERATURE = "75", GAS = '100',  HUMIDITY = '50', PRESSURE = '1000', ALTITUDE = '1000')

# Use sesion objec to comunicat to db
session = Session(bind=engine)

# Add test to the current session
session.add(test)

# Commit test to the database
session.commit()


