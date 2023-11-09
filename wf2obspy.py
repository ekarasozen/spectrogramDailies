# Created 2023-03-29 by Luke Underwood and Gabe Paris
# based heavily on db2stream from LT_gaps.py written by Steve Holtkamp
# module for AEC researchers to import waveform data from local servers to an obspy stream

import os
import sys

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

sys.path.append(os.environ['ANTELOPE'] + "/data/python")

import antelope.datascope as ds
from obspy import Stream, Trace
import numpy as np
import numpy.ma as ma


# inputs should be station, loc_code, channel, start time, and end time
# accepts * and ? wildcards
# returns obspy stream object with all relevant traces
def get_waveforms(network, station, location, channel, starttime, endtime):
            
    # functions placed inside of get_waveforms because they should not be exposed as a public interface for this module

    # Code that actually clears a trace object from memory, because ds.free() doesn't work
    def free_tr(tr):
        tr.record = ds.dbALL
        tr.table = ds.dbALL
        tr.trfree()

    # handles input given as a single string, translating into list format
    def interpret_input(input):
        if type(input) == str:
            # turn it into a list, dividing on commas
            input = input.split(',')

            # remove any whitespace
            for i, string in enumerate(input):
                input[i] = string.strip()

        return input

    # translates wildcards into antelope's atypical format
    def replace_wildcard(input):
        input = input.replace('*', '.*')
        input = input.replace('?', '.')
        return input

    # opens the database and returns a join of wfdisc and snetsta tables
    def open_db(time):
        ym = time.strftime('%Y_%m')
        ymd = ym + time.strftime('_%d')
        db_nme = f'/aec/db/waveforms/{ym}/waveforms_{ymd}'
        try:
            database = ds.dbopen(db_nme, 'r')
            wfdisc = database.lookup(table='wfdisc')
            wfdisc = wfdisc.join("snetsta")
        except Exception as e:
            print("Problem loading the database [%s] for processing!" % db_nme)
            raise e
        return wfdisc

    if starttime >= endtime:
        raise ValueError(f"Invalid times. The first time ({str(starttime)}) must be before the second ({str(endtime)})")

    # empty Stream object for result
    st = Stream()
    # used to validate traces. chanids[i] corresponds to st[i]
    chanids = []
    
    # create a list of databases spanning the full date range
    databases = [open_db(starttime)]
    curr_day = starttime
    while(curr_day.strftime("%D") != endtime.strftime("%D")):
        curr_day += 60*60*24
        databases.append(open_db(curr_day))

    # handle input
    network = interpret_input(network)
    station = interpret_input(station)
    location = interpret_input(location)
    channel = interpret_input(channel)
    

    curr_day = starttime
    # iterate over all days in databases array
    for wfdisc in databases:

        # get date strings formatted for trload
        e1 = curr_day.strftime("%D %H:%M:%S")
        if curr_day.strftime("%D") != endtime.strftime("%D"):
            e2 = curr_day.strftime("%D 23:59:59.98")
        else:
            e2 = endtime.strftime("%D %H:%M:%S")

        # iterate over all net-sta-loc-chan combos
        for n in network:
            for s in station:
                for l in location:
                    for c in channel:
                        net = replace_wildcard(n)
                        sta = replace_wildcard(s)
                        chan = replace_wildcard(c)
                        loc = replace_wildcard(l)
                        # print("Start of nested loops: ", net, sta, chan, loc)

                        # join channel and loc_code to match antelope format, handling wildcards appropriately
                        chan_str = chan
                        if loc == '' or loc == '.*':
                            chan_str += '.*'
                        else:
                            chan_str += "_" + loc

                        # string to subset for current net-sta-chan-loc combo
                        subset_str = f"snet =~ /{net}/ && sta =~ /{sta}/ && chan =~ /{chan_str}/"

                        # create an empty datascope trace
                        tr = ds.dbinvalid()

                        # subset to current net-sta-chan-loc
                        with ds.freeing(wfdisc.subset(subset_str)) as db:

                            # load the waveforms
                            try:
                                tr = db.trload_css(e1, e2)
                            except:
                                # it was decided that simply not including the trace is preferable
                                continue
                            
                            # Iterate over the trace object
                            for t in tr.iter_record():

                                # get metadata from trace object
                                nsamp, samprate = t.getv('nsamp', 'samprate')
                                sta, chan = t.getv('sta', 'chan')

                                # get the real network code
                                with ds.freeing(db.subset(f"sta == '{sta}' && chan == '{chan}'")) as net_subset:
                                    net_subset.record = 0
                                    net = net_subset.getv('snet')[0]
                                    chanid = net_subset.getv('chanid')[0]

                                # parse antelope chan_loc format
                                chan_split = chan.split('_')
                                chan = chan_split[0]
                                loc = chan_split[1] if len(chan_split) > 1 else ''

                                # get the pre-existing trace from stream object if it exists, otherwise create a new trace
                                # (the trace would already exist if there are multiple dbs in databases)
                                if chanid not in chanids:
                                    tr0 = Trace()
                                    tr0.stats.network = net
                                    tr0.stats.station = sta
                                    tr0.stats.channel = chan
                                    tr0.stats.location = loc
                                    tr0.stats.sampling_rate = samprate
                                    tr0.stats.npts += nsamp
                                    tr0.stats.starttime = starttime
                                elif curr_day == starttime:
                                    raise RuntimeError(f"There are duplicate waveforms for {net} {sta} {chan} {loc} {curr_day.strftime('%D')}")
                                else:
                                    ind = chanids.index(chanid)
                                    tr0 = st[ind]

                                d = np.array(t.trdata()) # get the actual data
                                d[abs(d)>=1e+30] = np.nan # Fix gaps
                                d_masked = ma.masked_invalid(d)

                                # Populate the trace object
                                
                                tr0.data = np.concatenate((tr0.data, d_masked))

                                # add the trace to the stream if there is data and it is not already in the stream
                                if len(tr0.data) > 0 and curr_day == starttime and chanid not in chanids:
                                    st.append(tr0)
                                    chanids.append(chanid)

                            free_tr(tr)

        # advance by one day, and set time to 00:00:00
        curr_day += 60*60*24
        curr_day = curr_day.replace(hour=0, minute=0, second=0)

    # Clean-up
    for database in databases:
        database.table = ds.dbALL
        database.close()
    
    return st.sort()
