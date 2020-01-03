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
    date_max = df_ims.RESOLUTION_DATE.max().tz_convert(None)
    return df_ims, date_min, date_max

def read_esmi():
    df = pd.read_pickle("../incidence_data/esmi_outage_db")
    # df['min_date_hour'] = df['min_date_hour'].apply(lambda x: pd.to_datetime(x))
    df['min_date_hour'] = df['min_date_hour'].dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    df['hour_esmi'] = df.min_date_hour.dt.hour
    return df

def hourly_counts(df, col):
    db = df[col].value_counts()
    db = db.reset_index()
    db = db.sort_values(by=['index'])
    return db

def outage_curves(db, col, plot_type, title):
    idx = np.arange(len(db))
    if plot_type == "bar":
        plt.bar(idx, db[col])
    else:
        plt.plot(idx, db[col])
    plt.xlabel("Time of day")
    plt.ylabel("Total Outages Detected")
    plt.title(title)
    plt.show()
    return None

def ims_esmi_combined(dims, desmi):
    di = dims.copy()
    de = desmi.copy()
    df = pd.merge(di, de, on=['index'])
    df.DETECTION_HOUR = (df.DETECTION_HOUR - df.DETECTION_HOUR.min()) * 100.0/(df.DETECTION_HOUR.max() - df.DETECTION_HOUR.min())
    df.hour_esmi = (df.hour_esmi - df.hour_esmi.min()) * 100.0/(df.hour_esmi.max() - df.hour_esmi.min())
    return df

def plot_combined(df):
    plt.plot(df.DETECTION_HOUR, label='IMS')
    plt.plot(df.hour_esmi, label='ESMI')
    # plt.legend(['IMS','ESMI'])
    plt.legend()
    plt.xlabel("Time of day")
    plt.ylabel("Normalized Outage Frequency (%)")
    plt.title("Outages Observed and Recorded by Time of Day")
    plt.show()
    return None

if __name__ == '__main__':
    tx_list = create_tx_list()
    # # setting filtering_by_tx = False will consider all the transformers
    df_ims, _, _ = read_ims(tx_list, filtering_by_tx=False)
    df_esmi = read_esmi()
    dh_ims = hourly_counts(df_ims, "DETECTION_HOUR")
    dh_esmi = hourly_counts(df_esmi, "hour_esmi")
    # outage_curves(dh_ims, "DETECTION_HOUR", "bar", "Outages by time of day (IMS Incidence 2017-18)")
    # outage_curves(dh_ims, "DETECTION_HOUR", "line", "Outages by time of day (IMS Incidence 2017-18)")
    # outage_curves(dh_esmi, "hour_esmi", "bar", "Outages by time of day (ESMI 2017-18)")
    # outage_curves(dh_esmi, "hour_esmi", "line", "Outages by time of day (ESMI 2017-18)")
    df_c = ims_esmi_combined(dh_ims, dh_esmi)
    plot_combined(df_c)
    code.interact(local = locals())
