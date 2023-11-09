#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:33:37 2023

@author: ezgikarasozen
"""

from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation

import datetime as dt


import functionsSpectrogram as specFunctions

#==========USER INPUT DATA==========#

#Date for the spectorgram plot
tday_str = "2023-11-08"
#tday_str = dt.datetime.now().strftime("%Y-%m-%d") #for today

# Data source for the station list
#dataSource = "AEC"
dataSource = "IRIS"


#===========CODES TO RUN, DO NOT CHANGE=========#
#Station list
# Barry Arm local station list
main_sta_list = ["BAT", "BAE"]
# supplemental regional station list
alt_sta_list = ["PWL", "KNK", "GLI"]

specFunctions.directoryManagement(tday_str)
#Grab station data 
if dataSource == "IRIS":
    data,sensitivity = specFunctions.grabIRISstationData(main_sta_list,alt_sta_list,tday_str)
if dataSource == "AEC":
    data,sensitivity = specFunctions.grabAECstationData(main_sta_list,alt_sta_list,tday_str)
#Plot spectrograms
specFunctions.plotSpectrogram(data,sensitivity,tday_str)  
#Create HTML table
specFunctions.runHTML(tday_str)  
print('Fin.')
#===================================#



