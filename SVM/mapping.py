#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:04:11 2019

@author: kellenbullock

GEOGRAPHY part


This file will be for making the 
coordiantes column, 
zipping latitude and longitude together, 
creating point geometry, 
reading in shapefile of ok counties,
setting coordinate system for both points, and counties,
displaying map.

"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# read in properties
properties = pd.read_csv('data.csv')

# reading in county boundries
ok_counties = gpd.read_file('Shapefile_Ok_Counties/tl_2016_40_cousub.shp', crs={'init': 'ESPG:2268'})

# ESPG:4326 WGS 1984
# ESPG:2268 NAD 1983
# creating geometry for properties
properties['Coordiantes'] =  list(zip(properties.Lat, properties.Long))
properties['Coordiantes'] = properties['Coordiantes'].apply(Point)
properties = gpd.GeoDataFrame(properties, crs={'init':'ESPG:2268'}, geometry='Coordiantes')
properties.plot()
#properties = properties.to_crs(ok_counties.crs)

#properties.plot()


# Loading in OK counties shapefile
#counties = gpd.GeoDataFrame('/Shapefile/COUNTY_BOUNDARY.shp', crs={'init': "SR-ORG:6703"})

# Map
f, ax = plt.subplots(1, figsize=(10,10))
ax = properties.plot(axes=ax, marker='*', markersize=8, legend=(False), color='red')
ok_counties.plot(ax=ax, edgecolor=('black'), color='blue')
ax.set_axis_off()
f.suptitle('Properties Distribution', fontsize=10)
plt.show()


'''
This is from the jupyter notebook using Geopandas to map the obesity points:

f, ax = plt.subplots(1, figsize=(16,12))
ax = data_plot.plot(axes=ax, column='Obesity',edgecolor=('black'), legend=(True))
parks.plot(ax=ax, marker='o', color='red', markersize=8)
gyms.plot(ax=ax, marker='o', color='Orange', markersize=8)
ax.set_axis_off()
f.suptitle('Map 3',fontsize=20)
lims = plt.axis('equal')
plt.show()

'''


'''
# I think I can largely ignore this:

def points(df, Lat, Long):
        
        This code is an example from another project to convert sparate lat longs to geometry
        from shapely.geometry import Point
        
        properites = pd.read_excel('Gym_Points.xlsx')
        gyms['Coordinates'] = list(zip(gyms.Long, gyms.Lat))
        gyms['Coordinates'] = gyms['Coordinates'].apply(Point)
        gyms = GeoDataFrame(gyms, crs={'init': 'epsg:4269'}, geometry='Coordinates')
        type(gyms)
        gyms.head()
        
        
        df['Coordinates'] = list(zip(df.Long, df.Lat))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        Properties = GeoDataFrame(df, crs={'init': 'epsg:4269'}, geometry='Coordinates')

    def plot():
        df = GeoDataFrame(df, crs={'init': 'espg:4269'}, geometry = 'Coordinates')
        return 
'''