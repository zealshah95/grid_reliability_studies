import pandas as pd
import geopandas as gpd
import numpy as np
import code
import datetime
import matplotlib.pyplot as plt

def create_tx_list():
    #############################################################################
    # Use this function to create a list of transformers to be used for analysis
    #############################################################################
    df_locs = gpd.read_file('./nairobi_20_tx_pts/nairobi_20_tx_pts.shp')
    tx_list = df_locs.INSTALATIO.unique()
    return tx_list

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
    df_ims["RESOLUTION_DT"] = df_ims.RESOLUTION_DATE.dt.date
    df_ims["DETECTION_HOUR"] = df_ims.DETECTION_DATE.dt.hour
    df_ims["RESOLUTION_HOUR"] = df_ims.RESOLUTION_DATE.dt.hour

    # Extract min and max datetime range
    date_min = df_ims.DETECTION_DATE.min().tz_convert(None)
    date_max = df_ims.DETECTION_DATE.max().tz_convert(None)
    return df_ims, date_min, date_max

def scatter_plot_util(dg, i, out_folder):
    dg1 = dg.loc[(dg.occurences != 0)]
    dg2 = dg.loc[(dg.occurences == 0)]
    if dg1.empty:
        None
    else: #Only plot graphs for tx with corresponding data points in NL data
        print("#########")
        print(i)
        print(len(dg1))
        print(len(dg2))
        dg1 = dg1.sort_values(by=['Scan_Time'])
        dg2 = dg2.sort_values(by=['Scan_Time'])
        print(dg1.Scan_Time)
        plt.figure()
        plt.scatter(dg2.Scan_Time, dg2.RadE9_Mult_Nadir_Norm, s=4, c='b', alpha=0.6)
        plt.scatter(dg1.Scan_Time, dg1.RadE9_Mult_Nadir_Norm, s=6, c='r')
        plt.savefig("./{}/{}_rad.png".format(out_folder,i))
    del(dg1)
    del(dg2)
    return None

def ims_nl_exploratory_analysis(df_ims, df_nl):
    #############################################################################
    # Scanning time according to df_nl is around 22, 23 hours every night
    #############################################################################
    scan_hours = df_nl.hour_nl.unique() #22, 23 for this dataset
    mean_scan_hour = scan_hours.mean() # 22.5
    # calculate the scanning buffer: time duration during which if an outage is
    # reported we expect to see some effect on radiance
    # create a new column in IMS data if that specific outage would lie within
    # the specified buffer range. Filter the dataframe keeping only the points
    # that lie within the buffer.
    # in_buffer field will be useful in visualizing as well.
    def inside_buffer(df):
        det_hour = df["DETECTION_HOUR"]
        # if 16<=det_hour<=23:
        if det_hour<=23:
            return "yes"
        else:
            return "no"
    df_ims["in_buffer"] = df_ims.apply(inside_buffer, axis=1)
    df_ims = df_ims[(df_ims.in_buffer != "no")]
    #calculate total number of outages recorded per installation per day within the given time period
    df_ims = df_ims.groupby(['INSTALATION','DETECTION_DT']).count()[['in_buffer']].reset_index()
    df_ims = df_ims.rename(columns = {'in_buffer':'occurences'})

    # create a new column in NL data which labels a reading as potential outage
    # recording using the new df_ims labels
    # This step may reduce the number of outages due to missing NL data row corresponding to that grid
    # on that date
    dnl = pd.merge(df_nl, df_ims[['INSTALATION','DETECTION_DT','occurences']], left_on=["Internal code", "date_nl"], right_on=['INSTALATION', 'DETECTION_DT'], how='left')

    # very less number of rows actually exhibit outage
    # at some points, the NL data is missing may be due to cloud based filtering
    # dnl_filt = dnl[(~dnl.in_buffer.isnull())]
    dnl.occurences = dnl.occurences.fillna(0)
    # code.interact(local = locals())

    for i in dnl["Internal code"].unique():
        dg = dnl[(dnl["Internal code"] == i)]
        # scatter_plot_util(dg, i, out_folder = "outputs_eve")
        scatter_plot_util(dg, i, out_folder = "outputs_fullday")
        del(dg)

    # code.interact(local = locals())
    return df_nl

if __name__ == '__main__':
    # tx_list = create_tx_list()
    tx_list = None
    # setting filtering_by_tx = False will consider all the transformers
    df_ims, dt_min, dt_max = read_ims(tx_list, filtering_by_tx=False)
    df_nl = read_nl(tx_list, dt_min, dt_max, filtering_by_tx=False, filtering_by_date=True)
    ims_nl_exploratory_analysis(df_ims, df_nl)
    code.interact(local = locals())
