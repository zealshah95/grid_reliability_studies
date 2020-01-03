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

def raw_plots(name, dg, type):
    # ,figsize=(16,8)
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(16,8))
    x = dg.date_nl.values
    mean_rad = dg.RadE9_Mult_Nadir_Norm.mean()
    mean_rad_arr = [mean_rad]*len(dg.date_nl)
    q1 = [dg.RadE9_Mult_Nadir_Norm.quantile([0.25]).values[0]]*len(dg.date_nl)
    q3 = [dg.RadE9_Mult_Nadir_Norm.quantile([0.75]).values[0]]*len(dg.date_nl)
    if type == "duration":
        s = dg.dur_sum.mean()
        s_arr = [s]*len(dg.date_nl)
        dg["color"] = dg.dur_sum.apply(lambda x: "b" if x<s else "r")
        # dgs = dg[(dg.dur_sum<s)]
        # dgb = dg[(dg.dur_sum>=s)]
        folder = "by_dur"
        ax2.bar(x, dg.dur_sum, label="Outage duration")
        ax2.plot(x, s_arr, label="Avg Outage duration", c="k")
    plt.legend()
    plt.grid()
    if dg.empty == False:
        # ax1.scatter(x, dgs.RadE9_Mult_Nadir_Norm, s=6, c="b")
        # ax1.scatter(x, dgb.RadE9_Mult_Nadir_Norm, s=6, c="r")
        ax1.scatter(x,dg.RadE9_Mult_Nadir_Norm, s=6, c=dg.color)
    # else:
    #     ax1.scatter(x, dgb.RadE9_Mult_Nadir_Norm, s=6, c="r")
    ax1.plot(x, mean_rad_arr, c="k", label="Average Rad")
    ax1.plot(x, q1, '--', c="g", label="Q1")
    ax1.plot(x, q3, '--', c="orange", label="Q2")
    plt.legend()
    plt.grid()
    plt.title("{}".format(name))
    plt.savefig("./outputs_ims_temporal/{}/{}.png".format(folder,name))
    return None

def stat_analysis(df, metric="mean", outages_threshold=True, filter_LI=False):
    """ Classify every NL point as lying above or below mean/median
    NL line for two cases: (1) all outages, (2) only above average
    outages """
    length_before = len(df)
    tx_before = df["Internal code"].nunique()
    #filter out all the points affected by lunar illuminance
    print("#####################################################")
    print("filter by LI: {}".format(filter_LI))
    print("mean thresholding: {}".format(outages_threshold))
    print("metric for rad: {}".format(metric))
    print("Length of df: {}".format(len(df)))

    #################################################
    # FILTERING STEP 1
    if filter_LI == True:
        df = df[(df.LI == 0)]
    print("df after LI: {}".format(len(df)))
    ################################################

    # calculate difference between radiance and corresponding mean radiance
    if metric == "mean":
        dist_from_metric = lambda df: df-df.mean()
    elif metric == "median":
        dist_from_metric = lambda df: df-df.median()
    else:
        raise ValueError("metric can only be mean or median")

    df["rad_position"] = df.groupby(["Internal code"])["RadE9_Mult_Nadir_Norm"].transform(dist_from_metric)
    df["rad_class"] = df.rad_position.apply(lambda x: "above_m" if x>0 else "below_m")

    #################################################
    # FILTERING STEP 2
    # Remove all the points with total outage duration = 0
    df = df.loc[(df.dur_sum!=0)]
    print("df after removing 0 duration outages: {}".format(len(df)))
    #################################################

    # calculate difference between outage duration
    # and mean outage duration. In this calculation we only
    # consider outages with total daily duration > 0.
    dur_from_mean = lambda dk: dk - dk.mean()
    df["outage_position"] = df.groupby(["Internal code"])["dur_sum"].transform(dur_from_mean)
    df["outage_class"] = df.outage_position.apply(lambda x: "above_mean" if x>=0 else "below_mean")

    #################################################
    # FILTERING STEP 3
    # if we just want to focus on outages that are above the mean outage duration
    if outages_threshold == True:
        df = df.loc[(df.outage_class == "above_mean")]
    print("df with only below mean outages: {}".format(len(df)))
    #################################################

    # Calculate mean/median number of outages per tx after all the filtering
    mean_faults_per_tx = df.groupby(['Internal code']).outage_position.value_counts().mean()
    median_faults_per_tx = df.groupby(['Internal code']).outage_position.value_counts().median()

    # Calculate average probability of detecting an outage using NL data i.e. below avg radiance
    # across all the transformers
    davg = df.groupby(["Internal code", "rad_class"]).count()[['rad_position']].reset_index()
    f = lambda df: df/df.sum()
    davg["probab"] = davg.groupby(["Internal code"]).rad_position.transform(f)
    avg_probab_outage = davg[(davg.rad_class == "below_m")].probab.mean()
    med_probab_outage = davg[(davg.rad_class == "below_m")].probab.median()
    print("Average Probability: {}".format(avg_probab_outage))
    print("Median Probability: {}".format(med_probab_outage))

    # Calculate total probability of detecting an outage using below avg rad across
    # all transformers
    above_m_count = df[(df.rad_class == "above_m")]["rad_position"].count()
    below_m_count = df[(df.rad_class == "below_m")]["rad_position"].count()
    total_probab_outage = below_m_count/(above_m_count + below_m_count)
    print("Overall Probability: {}".format(total_probab_outage))

    length_after = len(df)
    tx_after = df["Internal code"].nunique()
    return avg_probab_outage, med_probab_outage, total_probab_outage, length_before, length_after, tx_before, tx_after, mean_faults_per_tx, median_faults_per_tx

def zscore_analysis(df, out_filename, outages_threshold=False, filter_LI=False):
    print("df before LI: {}".format(len(df)))
    #################################################
    # FILTERING STEP 1
    if filter_LI == True:
        df = df[(df.LI == 0)]
    print("df after LI: {}".format(len(df)))
    ################################################
    f = lambda dk: (dk-dk.mean())/dk.std()
    df["z_score"] = df.groupby(["Internal code"])["RadE9_Mult_Nadir_Norm"].transform(f)
    db_with_outage = df.groupby(["Internal code"]).mean()[["z_score"]].reset_index()
    db_with_outage = db_with_outage.rename(columns = {'z_score':'z_o'})

    db_no_outage = df[(df.dur_sum==0)].groupby(["Internal code"]).mean()[["z_score"]].reset_index()
    db_no_outage = db_no_outage.rename(columns = {'z_score':'z_no'})

    #################################################
    # FILTERING STEP 2
    # Remove all the points with total outage duration = 0
    db = df.loc[(df.dur_sum!=0)]
    print("df after removing 0 duration outages: {}".format(len(db)))
    #################################################

    # calculate difference between outage duration
    # and mean outage duration. In this calculation we only
    # consider outages with total daily duration > 0.
    dur_from_mean = lambda dk: dk - dk.mean()
    db["outage_position"] = db.groupby(["Internal code"])["dur_sum"].transform(dur_from_mean)
    db["outage_class"] = db.outage_position.apply(lambda x: "above_mean" if x>=0 else "below_mean")
    # code.interact(local = locals())
    #################################################
    # FILTERING STEP 3
    # if we just want to focus on outages that are above the mean outage duration
    if outages_threshold == True:
        db = db.loc[(db.outage_class == "above_mean")]
    print("df with only above mean outages: {}".format(len(db)))
    #################################################

    db_only_outage = db.groupby(["Internal code"]).mean()[["z_score"]].reset_index()
    db_only_outage = db_only_outage.rename(columns = {'z_score':'z_on'})

    dz = db_with_outage.merge(db_no_outage, on=["Internal code"]).merge(db_only_outage, on=["Internal code"])
    z_overall = dz.mean()["z_o"]
    z_noout = dz.mean()["z_no"]
    z_onlyout = dz.mean()["z_on"]

    #################################################
    # Plot z-scores
    #################################################
    plt.figure(figsize=(16,6))
    plt.scatter(dz.index, dz.z_on, s=5, c='g', label="Only Outages")
    plt.scatter(dz.index, dz.z_no, s=5, c='b', marker='D',label="Overall without outages")
    plt.scatter(dz.index, dz.z_o, s=5, c='r', marker='^',label="Overall with outages")
    plt.legend()
    plt.xlabel("Transformer ID")
    plt.ylabel("Z-Score")
    plt.title("Z score for every transformer")
    plt.savefig("./outputs_ims_temporal/z_score/{}.png".format(out_filename))
    return z_overall, z_noout, z_onlyout

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

    #############################################################################
    # Process IMS and aggregate with NL data
    #############################################################################
    # Processed IMS will contain daily outage duration, mean and count
    dims = process_ims(df_ims)
    # Combine aggregated IMS data and NL data
    dc = pd.merge(df_nl, dims, left_on=['Internal code','date_nl'], right_on=['INSTALATION','DETECTION_DT'], how="left")
    dc = dc.fillna(0)

    #############################################################################
    # Plot aggregated data
    #############################################################################
    # for i in dc["Internal code"].unique():
    #     raw_plots(i, dc[(dc["Internal code"]==i)].sort_values(by=['date_nl']),"duration")

    #############################################################################
    # Statistical distribution of NL wrt outages
    #############################################################################
    do = pd.DataFrame(columns = {'filter_LI','mean_outage_thresholding','radiance_dist_metric','mean_probab','median_probab','overall_probab','len_before','len_after','tx_before','tx_after', 'mean_outages_per_tx', 'median_outages_per_tx'})

    for li in [True,False]:
        for out_thres in [True,False]:
            for rad_metric in ["mean","median"]:
                a,m,o,lb,la,tb,ta,mo,medo = stat_analysis(dc, metric=rad_metric, outages_threshold=out_thres, filter_LI=li)
                do = do.append({'filter_LI':li, 'mean_outage_thresholding': out_thres, 'radiance_dist_metric': rad_metric,
                               'mean_probab':a, 'median_probab':m, 'overall_probab':o, 'len_before':lb, 'len_after':la,
                               'tx_before':tb, 'tx_after':ta, 'mean_outages_per_tx':mo, 'median_outages_per_tx':medo}, ignore_index=True)
    do = do[['filter_LI','mean_outage_thresholding','radiance_dist_metric','mean_probab','median_probab','overall_probab','len_before','len_after','tx_before','tx_after','mean_outages_per_tx','median_outages_per_tx']]

    #############################################################################
    # Z-score of NL with and without outages
    #############################################################################
    # dz = zscore_analysis(dc, outages_threshold=False, filter_LI=False)
    dk = pd.DataFrame(columns = {'filter_LI', 'mean_outage_thresholding', 'overall', 'overall_no_outages', 'only_outages'})
    for li in [True, False]:
        for out_thres in [True, False]:
            zo, zno, zon = zscore_analysis(dc, "{}-li_{}-thres".format(li, out_thres), outages_threshold=out_thres, filter_LI=li)
            dk = dk.append({'filter_LI':li, 'mean_outage_thresholding':out_thres, 'overall':zo, 'overall_no_outages':zno, 'only_outages':zon}, ignore_index=True)
    dk = dk[['filter_LI', 'mean_outage_thresholding', 'overall', 'overall_no_outages', 'only_outages']]
    code.interact(local = locals())