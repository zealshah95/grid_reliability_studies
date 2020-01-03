import pandas as pd
import geopandas as gpd
import numpy as np
import code
import pickle
import glob
import datetime
import matplotlib.pyplot as plt

def create_tx_list():
    #############################################################################
    # Use this function to create a list of transformers to be used for analysis
    #############################################################################
    df_locs = gpd.read_file('./nairobi_20_tx_pts/nairobi_20_tx_pts.shp')
    tx_list = df_locs.INSTALATIO.unique()
    return tx_list

def read_ims(tx_codes_list, filtering_by_tx=True):
    #############################################################################
    # Input: - List of transformer Internal codes
    # Output: - Dataframe containing outage data corresponding to the listed
    # transformers.
    #         - date range of outages
    #############################################################################
    df_17 = pd.read_excel('../incidence_data/incidences2017.xlsx')
    df_18 = pd.read_excel('../incidence_data/incidences2018.xlsx')
    df_ims = pd.concat([df_17, df_18])

    if filtering_by_tx == True: #initial limited 20 datapoints
        df_ims = df_ims[(df_ims.INSTALATION.isin(tx_codes_list))]

    # Convert timestamps to pandas dt and change the timezone to UTC
    df_ims.DETECTION_DATE = df_ims.DETECTION_DATE.apply(lambda x: pd.to_datetime(x))
    df_ims.RESOLUTION_DATE = df_ims.RESOLUTION_DATE.apply(lambda x: pd.to_datetime(x))
    df_ims.DETECTION_DATE = df_ims.DETECTION_DATE.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    df_ims.RESOLUTION_DATE = df_ims.RESOLUTION_DATE.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    df_ims["DETECTION_DT"] = df_ims.DETECTION_DATE.dt.date
    # df_ims["RESOLUTION_DT"] = df_ims.RESOLUTION_DATE.dt.date
    df_ims["DETECTION_HOUR"] = df_ims.DETECTION_DATE.dt.hour
    # df_ims["RESOLUTION_HOUR"] = df_ims.RESOLUTION_DATE.dt.hour

    #remove all points with detection/resolution year 2016
    df_ims = df_ims[(~df_ims.DETECTION_DATE.dt.year.isin([2016,2019]))]

    # Extract min and max datetime range
    date_min = df_ims.DETECTION_DATE.min().tz_convert(None)
    date_max = df_ims.DETECTION_DATE.max().tz_convert(None)
    return df_ims, date_min, date_max

def read_nl(tx_codes_list, date_from, date_to, filtering_by_tx = False, filtering_by_date=False):
    #############################################################################
    # Input: - List of transformer Internal codes
    #        - Date range for filtering by date
    # Output: Dataframe containing NL data of grid cells containing the listed
    # transformers.
    #############################################################################
    # Read file that links transformer codes with grid cells in NL data of Nairobi
    tx_nl_file = pd.read_excel('./tx_locs_kenya_profile_match.xlsx')
    if filtering_by_tx == True:
        tx_nl_file = tx_nl_file[(tx_nl_file["Internal code"].isin(tx_codes_list))]

    # Append a dictionary with keys as Tx codes and data as corresponding NL dataframe
    dic = {}
    for i in tx_nl_file["Internal code"].unique():
        try:
            df = tx_nl_file[(tx_nl_file["Internal code"]==i)]
            x = int(df["Nairobi_grid_x"].values[0])
            y = int(df["Nairobi_grid_y"].values[0])
            dic[i] = pd.read_csv("./nairobi_grid_cf0li_corr/nairobi_grid_{}_{}.rm_adj.cf0li_corr.csv".format(x,y))
        except:
            continue
    # Add a new column to each dataframe containing corresponding Tx codes
    # combine all the dataframes to one
    arr = []
    for key, df in dic.items():
        df['Internal code'] = key
        arr.append(df)
    dn = pd.concat(arr)

    # Pre-process the dataframe
    dn.Scan_Time = dn.Scan_Time.apply(lambda x: pd.to_datetime(x))
    dn['date_nl'] = dn.Scan_Time.dt.date
    dn['hour_nl'] = dn.Scan_Time.dt.hour
    dn['minute_nl'] = dn.Scan_Time.dt.minute
    dn = dn.drop_duplicates()

    # code.interact(local = locals())
    # Filter data by datetime range
    if filtering_by_date == True:
        dn = dn[(dn.Scan_Time<=date_to) & (dn.Scan_Time>=date_from)]
    return dn

def process_ims(df):
    dg = df.groupby(['INSTALATION', pd.Grouper(key='DETECTION_DATE', freq='1D')]).aggregate({'SR_DURATION':['sum','mean','count'],'AFFECTED_CUSTOMERS':['sum','mean']}).reset_index()
    dg.columns = ['INSTALATION','DETECTION_DATE','dur_sum','dur_mean','dur_count','cust_sum','cust_mean']
    dg = dg.fillna(0)
    dg['DETECTION_DT'] = dg.DETECTION_DATE.dt.date
    return dg

def kurtosis_skew_mean(df, time_freq):
    df = df[(df.LI==0)] # filter by LI
    dg_kurt = df.groupby(["id","Latitude","Longitude",pd.Grouper(key="Scan_Time",freq=time_freq)])[['RadE9_Mult_Nadir_Norm']].apply(pd.DataFrame.kurtosis).reset_index()
    dg_kurt = dg_kurt.rename(columns = {'RadE9_Mult_Nadir_Norm':'rad_kurt'})
    dg_rest = df.groupby(["id","Latitude","Longitude",pd.Grouper(key="Scan_Time",freq=time_freq)]).aggregate({'RadE9_Mult_Nadir_Norm':['mean','skew']}).reset_index()
    dg_rest.columns = ["id","Latitude","Longitude","Scan_Time","rad_mean","rad_skew"]
    dg = pd.merge(dg_kurt,dg_rest,on=["id","Latitude","Longitude","Scan_Time"])
    code.interact(local = locals())
    return None

if __name__ == '__main__':
    # tx_list = create_tx_list()
    tx_list = None
    #############################################################################
    # two of the raw primary datasets have been saved as pickle files
    #############################################################################
    # By setting filtering_by_tx = False will consider all the transformers
    # df_ims, dt_min, dt_max = read_ims(tx_list, filtering_by_tx=False)
    # df_nl = read_nl(tx_list, dt_min, dt_max, filtering_by_tx=False, filtering_by_date=True)
    df_ims = pd.read_pickle("df_ims.pck")
    df_nl = pd.read_pickle("df_nl.pck")

    kurtosis_skew_mean(df_nl, "1Y")
    #############################################################################
    # Process IMS and aggregate with NL data
    #############################################################################
    # Processed IMS will contain daily outage duration, mean and count
    dims = process_ims(df_ims)
    # Combine aggregated IMS data and NL data
    dc = pd.merge(df_nl, dims, left_on=['Internal code','date_nl'], right_on=['INSTALATION','DETECTION_DT'], how="left")
    dc = dc.fillna(0)

    code.interact(local = locals())