"""
Spatial analysis of NL data at FBBU level.

It plots NL profiles and colors any radiance point
corresponding to an outage recorded in it's associated
FBBU red.

Idea is to see if non-aggregated FBBU outages
can be observed in the lower part of DNB profiles
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import code
import pickle
import datetime
import matplotlib.pyplot as plt

def read_nl():
    #############################################################################
    # Input: - List of transformer Internal codes
    #        - Date range for filtering by date
    # Output: Dataframe containing NL data of grid cells containing the listed
    # transformers.
    #############################################################################
    # Read file that links FBBUs with grid cells in NL data of Nairobi
    fbbu_nl_file = pickle.load(open("fbbu_to_nl.pck",'rb'))

    # Append a dictionary with keys as FBBU names and data as corresponding NL dataframe
    dic = {}
    for i in fbbu_nl_file.keys():
        temp = []
        for nl in fbbu_nl_file[i]:
            try:
                temp.append(pd.read_csv("./nairobi_grid_cf0li_corr/{}.rm_adj.cf0li_corr.csv".format(nl)))
            except:
                print("lost due to cloud based filtering")
        dic[i] = pd.concat(temp)

    # Add a new column to each dataframe containing corresponding FBBU names
    # combine all the dataframes to one
    arr = []
    for key, df in dic.items():
        df['fbbu_name'] = key
        arr.append(df)
    dn = pd.concat(arr)

    code.interact(local = locals())
    # Pre-process the dataframe
    dn.Scan_Time = dn.Scan_Time.apply(lambda x: pd.to_datetime(x))
    dn['date_nl'] = dn.Scan_Time.dt.date
    dn['hour_nl'] = dn.Scan_Time.dt.hour
    dn['minute_nl'] = dn.Scan_Time.dt.minute
    dn = dn.drop_duplicates()
    # code.interact(local = locals())
    # Filter data by datetime range
    return dn

def read_ims():
    #############################################################################
    # Input: - List of transformer Internal codes
    # Output: - Dataframe containing outage data corresponding to the listed
    # transformers.
    #         - date range of outages
    #############################################################################
    df_17 = pd.read_excel('../incidence_data/incidences2017.xlsx')
    df_18 = pd.read_excel('../incidence_data/incidences2018.xlsx')
    df_ims = pd.concat([df_17, df_18])

    # Convert timestamps to pandas dt and change the timezone to UTC
    df_ims.DETECTION_DATE = df_ims.DETECTION_DATE.apply(lambda x: pd.to_datetime(x))
    df_ims.RESOLUTION_DATE = df_ims.RESOLUTION_DATE.apply(lambda x: pd.to_datetime(x))
    df_ims.DETECTION_DATE = df_ims.DETECTION_DATE.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    df_ims.RESOLUTION_DATE = df_ims.RESOLUTION_DATE.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    df_ims["DETECTION_DT"] = df_ims.DETECTION_DATE.dt.date
    df_ims["DETECTION_HOUR"] = df_ims.DETECTION_DATE.dt.hour

    # Extract min and max datetime range
    # date_min = df_ims.DETECTION_DATE.min().tz_convert(None)
    # date_max = df_ims.DETECTION_DATE.max().tz_convert(None)

    date_min = df_ims.DETECTION_DT.min()
    date_max = df_ims.DETECTION_DT.max()
    return df_ims, date_min, date_max

def scatter_plot_util(dg, idn, fbbu, out_folder):
    if fbbu == "DANDORA/KARIOBANGI":
        fbbu = "DANDORA-KARIOBANGI"
    elif fbbu == "EMALI/SULTAN HAMUD":
        fbbu = "EMALI-SULTAN HAMUD"
    elif fbbu == "KIBWEZI/MTITO ANDEI":
        fbbu = "KIBWEZI-MTITO ANDEI"
    elif fbbu == "MANDERA/EL WAK":
        fbbu = "MANDERA-EL WAK"
    elif fbbu == "NAIROBI CBD/EASTLANDS":
        fbbu = "NAIROBI CBD-EASTLANDS"
    elif fbbu == "TALA/MATUU":
        fbbu = "TALA-MATUU"
    elif fbbu == "WESTLANDS/MUTHAIGA":
        fbbu = "WESTLANDS-MUTHAIGA"
    elif fbbu == "WOTE/MBUMBUNI":
        fbbu = "WOTE-MBUMBUNI"
    elif fbbu == "Parklands/Kitsuru":
        fbbu = "Parklands-Kitsuru"
    elif fbbu == "Jericho/Buruburu":
        fbbu = "Jericho-Buruburu"
    elif fbbu == "Kariobangi/Mwiki":
        fbbu = "Kariobangi-Mwiki"
    elif fbbu == "WAJIR/HABASWENI":
        fbbu = "WAJIR-HABASWENI"

    dg1 = dg.loc[(~dg.occurences.isnull())] #has outages
    dg2 = dg.loc[(dg.occurences.isnull())] #no outages
    if dg1.empty:
        None
    else: #Only plot graphs for tx with corresponding data points in NL data
        print("#########")
        print("{} & {}".format(fbbu,idn))
        print(len(dg1))
        print(len(dg2))
        # code.interact(local = locals())
        dg1 = dg1.sort_values(by=['date_nl'])
        dg2 = dg2.sort_values(by=['date_nl'])
        plt.figure(figsize=(16,6))
        plt.scatter(dg2.date_nl, dg2.RadE9_Mult_Nadir_Norm, s=4, c='b', alpha=0.6)
        plt.scatter(dg1.date_nl, dg1.RadE9_Mult_Nadir_Norm, s=6, c='r')
        plt.savefig("./{}/{}_{}_rad.png".format(out_folder,fbbu,idn))
    del(dg1)
    del(dg2)
    return None

def higher_spatial_analysis(dn, di, date_from, date_to):
    # filter nl data by IMS dates
    dn = dn[(dn.Scan_Time.dt.date<=date_to) & (dn.Scan_Time.dt.date>=date_from)]
    # code.interact(local = locals())
    # filter all IMS readings that were captured between 18:00 and 23:00 hours
    di = di[(di.DETECTION_HOUR<=23) & (di.DETECTION_HOUR>=20)]
    """combine IMS to NL such that each row of df_nl has an indicator associated
    if there was an outage recorded on that day in that specific FBBU"""
    dm = pd.merge(dn[['id','date_nl','fbbu_name','RadE9_Mult_Nadir_Norm']], di[['DETECTION_DT','FBBU_NAME']].drop_duplicates(), left_on=['fbbu_name','date_nl'], right_on=['FBBU_NAME','DETECTION_DT'], how="left")
    dm = dm.drop(columns=['FBBU_NAME'])
    dm = dm.rename(columns = {'DETECTION_DT':'occurences'})
    """Here we use the nl_to_fbbu pickle file which uses nl id as key and fbbu name
    as item"""
    nl_to_fbbu = pickle.load(open("nl_to_fbbu.pck",'rb'))
    j = 0
    for key in nl_to_fbbu.keys():
        idx = key #nl grid cell id
        fb = nl_to_fbbu[key] #fbbu
        print("{}_{}".format(fb,idx))
        dg = dm[(dm.id == idx) & (dm.fbbu_name == fb)]
        scatter_plot_util(dg, idx, fb, out_folder = "outputs_fbbu_spatial")
        del(dg)
        # j = j+1
        # if j == 10:
        #     break
            # print("Not present in dm")
    return None

if __name__ == '__main__':
    df_ims, dt_min, dt_max = read_ims()

    ###########
    """ df_nl is created in such a way that it contains FBBU name associated with
    every grid cell. One grid cell can belong to multiple FBBU areas. df_nl
    contains the complete NL profile for every grid cell. The profile might be redundant
    and repeated if a grid cell belongs to multiple FBBU. For each fbbu name and grid
    cell, we have one full profile from 2012 to 2019"""
    ###########
    # df_nl = read_nl(dt_min, dt_max, filtering_by_date=True) #saved as pickle
    df_nl = pd.read_pickle("fbbu_spatial_temp.pck") #directly read pickle file to save time
    higher_spatial_analysis(df_nl, df_ims, dt_min, dt_max)