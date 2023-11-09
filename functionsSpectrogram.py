#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mewest, sknoel
"""
import os
import sys
import shutil

import numpy as np

from obspy.clients.fdsn import Client
client_wm = Client("IRIS")
from obspy.clients.iris import Client  #this is needed for gc_distaz calculation
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

from scipy import signal

# Surpress log10 divide by zero warning raised by data gaps
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

# Directoy management, to overwrite or not to overwrite

def directoryManagement(tday_str,overwrite=True):
    tday      = UTCDateTime(tday_str)
    if overwrite == True: # overwire conflicting directories
        try:
            os.mkdir(tday.strftime("%Y%m%d"))
        except FileExistsError:
            shutil.rmtree(tday.strftime("%Y%m%d"), ignore_errors=True)
            os.mkdir(tday.strftime("%Y%m%d"))

    elif overwrite == False:  
        try:
            os.mkdir(tday.strftime("%Y%m%d"))
        except FileExistsError:        
            print('\nDirectory: ', tday.strftime("%Y%m%d"), ' already exists in', os.getcwd(), '\n')
            sys.exit('Existing directory has been retained.\nNo new spectograms created.\n\nEXITING GRACEFULLY.')


# Spectrogram Plotting Function
def plotSpectrogram(data, sensitivity,tday_str):
    tday      = UTCDateTime(tday_str)

    #===================================#
    # Start plotting
    print('\ncreating figure panels ...\n' )
    panelsperday = 144   # adjust 
    interval = 86400/panelsperday 
    for n in range(panelsperday):
        t1 = tday + n*interval
        t2 = tday + (n+1)*interval
        
        # GENERATE FIGURE  
        fig, ax = plt.subplots(6 ,1,figsize=(14,14))   
        
        xtickvalues = mdates.date2num([t1.datetime]) + np.arange(start=0,stop=600,step=60)/86400
        date_format = mdates.DateFormatter('%H:%M')
        vscale = 0.000003
        thresh = 0.0000014
        
        even_ax_position = [[.05, .80, .7, .10], [.05, .50, .7, .10], [.05, .20, .7, .10]]
        odd_ax_position  = [[.05, .65, .7, .15], [.05, .35, .7, .15], [.05, .05, .7, .15]]
        
        # Loop through data and add spectogram track (few as 1, many as 3)
        i = 0
        j = 0
        for station in data:
    
            station = station.slice(t1,t2,nearest_sample=True)
            station.detrend("linear")
            station.detrend("demean")
            sttmp = station[0].copy()
            sttmp.filter('highpass',freq=5.0)
    
            # AXIS TOP (even)
            ax[i].plot(sttmp.times("matplotlib"),sttmp.data/sensitivity[j][1],'-',color='k',label=station[0].stats.station,lw=.2)
            
            
            if i == 0:
                ax[i].set_title(t1.datetime.strftime("%Y-%m-%d         %H:%M") + " - " + t2.datetime.strftime("%H:%M"),fontsize=16)
            
            ax[i].tick_params(direction='in')
            ax[i].axes.xaxis.set_ticklabels([])
            ax[i].set_xlim(mdates.date2num(t1.datetime), mdates.date2num(t2.datetime)) 
            ax[i].set_xticks(xtickvalues)
            ax[i].set_yticks([])
            ax[i].set_ylim(-1*vscale,1*vscale)
            ax[i].spines['top'].set_color('white')
            ax[i].spines['bottom'].set_color('white')
            ax[i].set_position(even_ax_position[j])
            i += 1
            
            # AXIS BOTTOM (odd)
#           fxx, txx, Sxx = signal.spectrogram(station[0].data/sensitivity[j][1], fs=50, mode='psd', 
#                                               nperseg=256, noverlap=128, scaling='density')   #    (this is the original)
            fxx, txx, Sxx = signal.spectrogram(station[0].data/sensitivity[j][1], fs=50, mode='psd', 
                                              nperseg=256, noverlap=224, scaling='density')    #   (more overlap)
#           fxx, txx, Sxx = signal.spectrogram(station[0].data/sensitivity[j][1], fs=50, mode='psd', 
#                                              nperseg=512, noverlap=128, scaling='density')   #    (longer window)
#           fxx, txx, Sxx = signal.spectrogram(station[0].data/sensitivity[j][1], fs=50, mode='psd', 
#                                              nperseg=128, noverlap=96, scaling='density')     #    (shorter window)


            Sxx = np.flipud(Sxx)
            Sxx = 10*np.log10(Sxx)
            halffreq = (fxx[1]-fxx[0])/2
            extent = (mdates.date2num(t1.datetime), mdates.date2num(t2.datetime), fxx[0]-halffreq, fxx[-1]+halffreq)
            ax[i].imshow(Sxx, extent=extent, aspect='auto', cmap='nipy_spectral', vmin=-180, vmax=-120) 
            ax[i].set_xticks(xtickvalues)
            ax[i].xaxis.set_major_formatter(date_format)
            text = (station[0].stats.station)+' '+(station[0].stats.channel)
            ax[i].text(0.88, 0.90, text, transform=ax[i].transAxes, fontsize=14, fontweight='bold', color='white', 
                       path_effects=[pe.withStroke(linewidth=0.8, foreground="darkslategray")])
            ax[i].set_position(odd_ax_position[j])
    
            i += 1
            j += 1
    
        # Save figure with all spectrogram tracks
        fig.savefig(t1.strftime("%Y%m%d/%Y%m%d_%H%M")+'.png', bbox_inches='tight', pad_inches = 0)
        plt.close('all')
        


# Empty Stream objects for BAE, BAT, and PWL
# Used for plotting purposes when full day of data is missing 
def No_Data_Streams(station,starttime,endtime):
    
    if station == "BAT":

        BAT_stats = Stats()
        BAT_values = {'network': 'AK',
                     'station': 'BAT',
                     'location': '',
                     'channel': 'BHZ',
                     'starttime': starttime,
                     'endtime': endtime,
                     'delta': 0.02,
                     'sampling_rate':50.0}
        
        BAT_stats.update(BAT_values)
        BAT_trace = Trace()
        BAT_trace.stats = BAT_stats
        samples = int(24 * 60 * 60 * BAT_trace.stats.sampling_rate)
        zero_trace = np.zeros(samples)
        BAT_trace.data = zero_trace
        BAT = Stream()
        BAT.append(BAT_trace)
        
        stream = BAT
        response = [0.2, 503960188.2279222] #UPDATED: 2023/08/14
    
    
    if station == "BAE":
        
        BAE_stats = Stats()
        BAE_values = {'network': 'AK',
                     'station': 'BAE',
                     'location': '',
                     'channel': 'BHZ',
                     'starttime': starttime,
                     'endtime': endtime,
                     'delta': 0.02,
                     'sampling_rate':50.0}
        
        BAE_stats.update(BAE_values)
        BAE_trace = Trace()
        BAE_trace.stats = BAE_stats
        samples = int(24 * 60 * 60 * BAE_trace.stats.sampling_rate)
        zero_trace = np.zeros(samples)
        BAE_trace.data = zero_trace
        BAE = Stream()
        BAE.append(BAE_trace)
        
        stream = BAE
        response = [0.3, 501762803.4026484] #UPDATED: 2023/08/14
    
    # Station PWL (Default if no supp. stations have data)
    if station == "PWL":

        PWL_stats = Stats()
        PWL_values = {'network': 'AK',
                     'station': 'BAE',
                     'location': '',
                     'channel': 'BHZ',
                     'starttime': starttime,
                     'endtime': endtime,
                     'delta': 0.02,
                     'sampling_rate':50.0}
        
        PWL_stats.update(PWL_values)
        PWL_trace = Trace()
        PWL_trace.stats = PWL_stats
        samples = int(24 * 60 * 60 * PWL_trace.stats.sampling_rate)
        zero_trace = np.zeros(samples)
        PWL_trace.data = zero_trace
        PWL = Stream()
        PWL.append(PWL_trace)
        
        stream = PWL
        response = [0.3, 501762803.4026484] #UPDATED: 2023/08/14

    return stream, response

#==========GRAB AEC STATION DATA========#

def grabAECstationData(main_sta_list,alt_sta_list,tday_str):
    import wf2obspy

    starttime_str = tday_str + " 00:00:00.000000"
    endtime_str   = tday_str + " 23:59:59.999999"
    tday      = UTCDateTime(tday_str)
    starttime = UTCDateTime(starttime_str)
    endtime   = UTCDateTime(endtime_str)

    # Instrument Data Scale Factor, Hardcoded, needed for AEC data grab
    print('\nstation instrument response is harcoded, last modification date: 2023/08/23 \n')

    STA_scalefactor_dict = { 'BAT' : [0.2, 503960188.2279222, '2023/08/23'],
                             'BAE' : [0.3, 501762803.4026484, '2023/08/23'],
                             'PWL' : [0.3, 501762803.4026484, '2023/08/23'],
                             'KNK' : [0.3, 501762803.4026484, '2023/08/23'],
                             'GLI' : [0.3, 501762803.4026484, '2023/08/23']}    
    print('\ngrabbing waveforms for ' + tday.strftime("%Y%m%d"), '\n')
    data = []
    sensitivity = []
    for station in main_sta_list:
        if station == "BAT":
            # BAT data handeling
            try:
                BAT = wf2obspy.get_waveforms("AK", "BAT", "*", "BHZ", starttime, endtime)
                sensitivity.append(STA_scalefactor_dict['BAT'])
                print('BAT instrument scale factor/calibration data updated: ', STA_scalefactor_dict['BAT'][2])
                
                # Merge traces if more than one returned
                if len(BAT) > 1:
                    BAT.merge()
                
                # Fill station outage gaps
                BAT[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
                np.nan_to_num(BAT[0].data, copy=False, nan=0.0)
                
                data.append(BAT)
            except:
                BAT_stream, BAT_response = No_Data_Streams("BAT")
                data.append(BAT_stream)
                sensitivity.append(BAT_response)
                print('No Data: BAT — preparing spectrograms without BAT')
        if station == "BAE":
            # BAE data handeling
            try:   
                BAE = wf2obspy.get_waveforms("AK", "BAE", "*", "BHZ", starttime, endtime)
                sensitivity.append(STA_scalefactor_dict['BAE'])
                print('BAE instrument scale factor/calibration data updated: ', STA_scalefactor_dict['BAE'][2])
                
                # Merge traces if more than one returned
                if len(BAE) > 1:
                    BAE.merge()
                
                # Fill station outage gaps
                BAE[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
                np.nan_to_num(BAE[0].data, copy=False, nan=0.0)
                
                data.append(BAE)
            except:
                BAE_stream, BAE_response = No_Data_Streams("BAE")
                data.append(BAE_stream)
                sensitivity.append(BAE_response)
                print('No Data: BAE — preparing spectrograms without BAE')
        
    # supplemental regional station data handeling
    for station in alt_sta_list:
        try:
            STA = wf2obspy.get_waveforms("AK", station, "*", "BHZ", starttime, endtime)
            
            try:
                sensitivity.append(STA_scalefactor_dict[station])
                print(station, 'instrument scale factor/calibration data updated: ', STA_scalefactor_dict[station][2])
                
            except:
                txt_warning = 'WARNING: ' + station + ' instrument scale factor/calibration data NOT implimented in "STA_scalefactor_dict".'
                txt_exception = 'EXCEPTION: PWL used as defulat. Add ' + station + ' instrument scale factor/calibration data to "STA_scalefactor_dict" for accurate data representation.'
                print(txt_warning)
                print(txt_exception)
                print('PWL instrument reponse updated: ', STA_scalefactor_dict['PWL'][2])
                sensitivity.append(STA_scalefactor_dict['PWL'])
            
            # Merge traces if more than one returned
            if len(STA) > 1:
                STA.merge()
            
            # Fill station outage gaps
            STA[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
            np.nan_to_num(STA[0].data, copy=False, nan=0.0)
            
            data.append(STA)
            break;
            
        except:
            if station != alt_sta_list[-1]:
                print('No Data: ', station, ' — attempting to pull data for next station in list.' )
            
            else:
                PWL_stream, PWL_response = No_Data_Streams("PWL")
                data.append(PWL_stream)
                sensitivity.append(PWL_response)
                print('No Data: ', station, ' — this is the final listed supplemental network station.', 
                                                 '\nWARNING: Update supplemental station list with additional stations.',
                                                 '\nProceeding — preparing spectrograms without regional station')
    return data, sensitivity
       


#==========GRAB IRIS STATION DATA========#

def grabIRISstationData(main_sta_list,alt_sta_list,tday_str):
    starttime_str = tday_str + " 00:00:00.000000"
    endtime_str   = tday_str + " 23:59:59.999999"
    tday      = UTCDateTime(tday_str)
    starttime = UTCDateTime(starttime_str)
    endtime   = UTCDateTime(endtime_str)
    print('\ngrabbing waveforms for ' + tday.strftime("%Y%m%d"), '\n')
    data = []
    sensitivity = []
    for station in main_sta_list:
        if station == "BAT":
            # BAT data handeling
            try:
                BAT = client_wm.get_waveforms("AK", "BAT", "*", "BHZ", starttime, endtime, attach_response=True)
                sensitivity.append(BAT[0].stats.response._get_overall_sensitivity_and_gain(output='VEL'))
                
                # Merge traces if more than one returned 
                if len(BAT) > 1:
                    BAT.merge()
                
                # Fill station outage gaps
                BAT[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
                data.append(BAT)
            except:
                BAT_stream, BAT_response = No_Data_Streams("BAT",starttime,endtime)
                data.append(BAT_stream)
                sensitivity.append(BAT_response)
                print('No Data: BAT — preparing spectrograms without BAT')
    
        if station == "BAE":
            # BAE data handeling
            try:   
                BAE = client_wm.get_waveforms("AK", "BAE", "*", "BHZ", starttime, endtime, attach_response=True)
                sensitivity.append(BAE[0].stats.response._get_overall_sensitivity_and_gain(output='VEL'))
                
                # Merge traces if more than one returned 
                if len(BAE) > 1:
                    BAE.merge()
                
                # Fill station outage gaps
                BAE[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
                data.append(BAE)
            except:
                BAE_stream, BAE_response = No_Data_Streams("BAE",starttime,endtime)
                data.append(BAE_stream)
                sensitivity.append(BAE_response)
                print('No Data: BAE — preparing spectrograms without BAE')

    # supplemental regional station data handeling
    for station in alt_sta_list:
        try:
            STA = client_wm.get_waveforms("AK", station, "*", "BHZ", starttime, endtime, attach_response=True)
            sensitivity.append(STA[0].stats.response._get_overall_sensitivity_and_gain(output='VEL'))
            
            # Merge traces if more than one returned 
            if len(STA) > 1:
                STA.merge()
            
            # Fill station outage gaps
            STA[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
            data.append(STA)
            break;
            
        except:
            if station != alt_sta_list[-1]:
                print('No Data: ', station, ' — attempting to pull data for next station in list.' )
            
            else:
                PWL_stream, PWL_response = No_Data_Streams("PWL")
                data.append(PWL_stream)
                sensitivity.append(PWL_response)
                print('No Data: ', station, ' — this is the final chosen supplemental network station.', 
                                                 '\nWARNING: Update supplemental station listing.',
                                             '\nProceeding — preparing spectrograms without regional station')


    return data, sensitivity



#=============Writes an html table that formats spectrogram PNGs for easy viewing ============#

def runHTML(tday_str):
    tday      = UTCDateTime(tday_str)    
    date=tday.strftime("%Y%m%d")

    row_labels = ['00:00-01:00', '01:00-02:00', '02:00-03:00', '03:00-04:00', '04:00-05:00',
                  '05:00-06:00', '06:00-07:00', '07:00-08:00', '08:00-09:00', '09:00-10:00',
                  '10:00-11:00', '11:00-12:00', '12:00-13:00', '13:00-14:00', '14:00-15:00',
                  '15:00-16:00', '16:00-17:00', '17:00-18:00', '18:00-19:00', '19:00-20:00',
                  '20:00-21:00', '21:00-22:00', '22:00-23:00', '23:00-00:00']
    
    time_stamps = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                   '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    
    
    # Construct HTML code block for local directories only
    html_header = """\
    <html>
    <style>
    
    body {background-color: #8DB2A1}
    
    table, th, td {border-bottom: 15px solid #E4963D; border-collapse: collapse}
    tr:nth-child(even) {background-color: rgba(209, 241, 226, 0.25)}
    th:nth-child(even),td:nth-child(even) {background-color: rgba(209, 241, 226, 0.25)}
    
    </style>
    <body>
    <h2>spectrogram table!!!</h2>
    <table style="width:100%">
    """
    
    html_spectrogram_tables = []
    for i in range(len(time_stamps)):
        html_data_table = """\
          <tr>
           <td>{row_labels}</td>
                	<td><a href="./{date}/{date}_{time_stamps}00.png"><img src="./{date}/{date}_{time_stamps}00.png" style="width:250px;height:312.5px;"></a></td>
                <td><a href="./{date}/{date}_{time_stamps}10.png"><img src="./{date}/{date}_{time_stamps}10.png" style="width:250px;height:312.5px;"></a></td>
                <td><a href="./{date}/{date}_{time_stamps}20.png"><img src="./{date}/{date}_{time_stamps}20.png" style="width:250px;height:312.5px;"></a></td>
                <td><a href="./{date}/{date}_{time_stamps}30.png"><img src="./{date}/{date}_{time_stamps}30.png" style="width:250px;height:312.5px;"></a></td>
                <td><a href="./{date}/{date}_{time_stamps}40.png"><img src="./{date}/{date}_{time_stamps}40.png" style="width:250px;height:312.5px;"></a></td>
                <td><a href="./{date}/{date}_{time_stamps}50.png"><img src="./{date}/{date}_{time_stamps}50.png" style="width:250px;height:312.5px;"></a></td>
          </tr>
        """.format(row_labels=row_labels[i], date=date, time_stamps=time_stamps[i])    
        html_spectrogram_tables.append(html_data_table)
    
    html_full_spectrogram_table = """\
      <tr>
       <td></td>
        	<td style="text-align: center"> :00 — :10</td>
            <td style="text-align: center"> :10 — :20</td>
            <td style="text-align: center"> :20 — :30</td>
            <td style="text-align: center"> :30 — :40</td>
            <td style="text-align: center"> :40 — :50</td>
            <td style="text-align: center"> :50 — :00</td>
      </tr>
    """
    
    for i in range(len(time_stamps)):
        html_full_spectrogram_table = html_full_spectrogram_table + html_spectrogram_tables[i]
    
    html_footer = """\
        </table>
        </body>
        </html>"""
    
    html_file = html_header + html_full_spectrogram_table + html_footer
    
    # Create HTML file in present working directory
    html_file_name = date + '_DataView' + '.html'
    
    f = open(html_file_name,'w')
    f.write(html_file)
    f.close()
    



