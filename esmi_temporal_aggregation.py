import pandas as pd
import geopandas as gpd
import numpy as np
import code
import pickle
import glob
import datetime
import matplotlib.pyplot as plt

def daily_composites(comp_path, min_esmi_dt, max_esmi_dt):
    df_em = []
    dc = pd.DataFrame()
    for i in glob.glob(comp_path + "/*cf_corr*"):
        df_em.append(pd.read_csv(i))
    dc = pd.concat(df_em, ignore_index=True)
    #################################################################################
    # Columns: ['id', 'Latitude', 'Longitude', 'Agg_Name', 'Line_M10', 'Sample_M10',
    # 'File_Vflag', 'QF_Vflag', 'Line_DNB', 'Sample_DNB', 'File_DNB',
    # 'RadE9_DNB', 'Scan_Time', 'LI', 'VCM_Flag', 'SatZ_Bin_N',
    # 'SatZ_Bin_Mdn', 'RadE9_Bin_1Q', 'RadE9_Bin_Mdn', 'RadE9_Bin_3Q',
    # 'Adjust_Mult_Nadir_Norm', 'SatZ', 'RadE9_Aggzone_Norm',
    # 'RadE9_Mult_Nadir_Norm', 'Time_Segment', 'rownum', 'linear_fill',
    # 'knn_mean', 'detrend', 'trend', 'seasonal', 'bright', 'dim',
    # 'bright_thr', 'dim_thr', 'Light_Type']
    #################################################################################
    # convert datetime entries to pandas datetime from string
    dc.Scan_Time = dc.Scan_Time.apply(lambda x: pd.to_datetime(x))
    dc['date_nl'] = dc.Scan_Time.dt.date
    dc['hour_nl'] = dc.Scan_Time.dt.hour
    dc['minute_nl'] = dc.Scan_Time.dt.minute
    print("#######################")
    print("Length of NL data before filtering: {}".format(len(dc)))

    # filter entries recorded before and after ESMI data date range
    dc = dc[(dc.Scan_Time.dt.date<=max_esmi_dt) & (dc.Scan_Time.dt.date>=min_esmi_dt)]
    print("Length of NL data after filtering: {}".format(len(dc)))
    print("#######################")
    # extracting only the columns we need
    # dc = dc[['id','Latitude','Longitude','Agg_Name','QF_Vflag','LI','Scan_Time','RadE9_DNB','RadE9_Mult_Nadir_Norm','Time_Segment']]
    return dc

def esmi_data():
    ###########################################################
    # Read and pre-process raw voltage data
    ###########################################################
    # We get total duration of outages from original voltage database
    re1 = '/Users/zealshah/Documents/reliability_studies/ESMI_Kenya/' #contains original data
    dd =  pd.read_pickle(re1 + 'esmi_kenya_column_db')
    dd = dd.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dd['v_class'] = dd.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")
    #convert Eastern African time to UTC
    dd['date_hour'] = dd.date_hour.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    #create columns for date, hours and minutes
    dd['date'] = dd.date_hour.dt.date
    dd['hour'] = dd.date_hour.dt.hour
    dd['minute'] = dd.date_hour.dt.minute
    #adding id column to be used as key for NL data (basically just adds _ between name and number)
    dd['id'] = dd.Location_Name.apply(lambda x: str(x.split(' ')[0] + '_' + x.split(' ')[1]))

    ###########################################################
    # Read and pre-process outage events data
    ###########################################################
    # We get count of outages from outage events data
    re2 = '/Users/zealshah/Documents/reliability_studies/old_nl/' #contains outage data
    dc = pd.read_pickle(re2 + 'voltage_events_esmi_kenya')
    dc.min_date_hour = dc.min_date_hour.dt.tz_localize("Africa/Nairobi").dt.tz_convert("UTC")
    dc.max_date_hour = dc.max_date_hour.dt.tz_localize("Africa/Nairobi").dt.tz_convert("UTC")
    dc = dc.drop(columns = ["event_date_str"])
    dc.event_date = pd.to_datetime(dc.event_date, format = "%Y-%m-%d")
    dc['id'] = dc.Location_Name.apply(lambda x: str(x.split(' ')[0] + '_' + x.split(' ')[1]))
    # code.interact(local = locals())

    min_esmi_dt = dd.date.min()
    max_esmi_dt = dd.date.max()
    return dd, dc, min_esmi_dt, max_esmi_dt

def daily_voltage_events(dd, dc):
    #create a table with site specific duration of different voltage event classes every month
    #we are considering only total duration of events over a day
    db_dur = dd.groupby(['id', 'v_class']).resample('1D', on='date_hour').count()[['date_hour']]
    db_dur = db_dur.rename(columns ={'date_hour':'event_dur'})
    db_dur.event_dur = db_dur.event_dur/60.0
    db_dur = db_dur.unstack(level = 1)
    db_dur.columns = db_dur.columns.droplevel(0)
    db_dur = db_dur.reset_index()

    #create a table with site specific count of different voltage event classes every month
    #we are considering only total actual count of events over the month
    db_ct = dc.groupby(['id', 'v_class']).resample('1D', on='min_date_hour').count()[['dur_hour']]
    db_ct = db_ct.unstack(level = 1)
    db_ct.columns = db_ct.columns.droplevel(0)
    db_ct = db_ct.reset_index()
    db_ct = db_ct.rename(columns = {'min_date_hour':'date_hour'})

    # merge both the dataframes
    db = pd.merge(db_dur, db_ct, how='left', left_on=['id', 'date_hour'], right_on = ['id', 'date_hour'])
    db = db.rename(columns = {'low_x':'low_dur', 'normal_x':'normal_dur', 'no_x':'no_dur', 'no_data_x':'no_data_dur', 'over_x':'over_dur', 'low_y':'low_ct', 'normal_y':'normal_ct', 'no_y':'no_ct', 'no_data_y':'no_data_ct', 'over_y':'over_ct'})
    db['ts_date'] = db.date_hour.apply(lambda x: x.date)
    db['ts_day'] = db.date_hour.apply(lambda x: x.day)
    db['ts_month'] = db.date_hour.apply(lambda x: x.month)
    db['ts_year'] = db.date_hour.apply(lambda x: x.year)
    # fill all the nans with zeros
    db = db.fillna(0)
    # combining low voltage and no supply
    db['uv_dur'] = db['low_dur'] + db['no_dur']
    db['uv_ct'] = db['low_ct'] + db['no_ct']
    # code.interact(local = locals())
    return db

def corr_db(db):
    db = db[['id','ts_date','low_dur', 'no_dur', 'no_data_dur', 'normal_dur',
       'over_dur', 'low_ct', 'no_ct', 'no_data_ct', 'normal_ct', 'over_ct',
       'uv_dur', 'uv_ct','RadE9_Mult_Nadir_Norm']]
    d_corr = db.groupby(['id']).corr().reset_index()
    d_corr = d_corr[(d_corr.level_1 == 'RadE9_Mult_Nadir_Norm')]
    # no data columns should not be used for correlation calculations
    d_corr.index = d_corr.id
    d_corr = d_corr.drop(columns = ['level_1', 'id', 'no_data_ct', 'no_data_dur','RadE9_Mult_Nadir_Norm'])
    d_corr['max_col'] = abs(d_corr).idxmax(axis = 1)
    # code.interact(local = locals())
    return d_corr

def raw_analysis(name, df, dnl, type):
    #[['id','date_nl','uv_dur','uv_ct','RadE9_Mult_Nadir_Norm']]
    plt.figure(figsize=(16,8))
    dg = df[(df.id == name)]
    dg = dg.fillna(0)
    mean_rad = dg.RadE9_Mult_Nadir_Norm.mean()
    mean_rad_arr = [mean_rad]*len(dg.date_nl)
    q1 = [dg.RadE9_Mult_Nadir_Norm.quantile([0.25]).values[0]]*len(dg.date_nl)
    q3 = [dg.RadE9_Mult_Nadir_Norm.quantile([0.75]).values[0]]*len(dg.date_nl)
    if type == "duration":
        s = dg.uv_dur.mean()*1.25
        s_arr = [s]*len(dg.date_nl)
        dgs = dg[(dg.uv_dur<s)]
        dgb = dg[(dg.uv_dur>=s)]
        # code.interact(local = locals())
        folder = "1.25mean-by_dur"
        plt.subplot(2,1,2)
        plt.bar(dg.date_nl, dg.uv_dur, label="UV duration")
        plt.plot(dg.date_nl, s_arr, label="Avg UV duration", c="k")
    else:
        s = dg.uv_ct.mean()*1.25
        s_arr = [s]*len(dg.date_nl)
        dgs = dg[(dg.uv_ct<s)]
        dgb = dg[(dg.uv_ct>=s)]
        folder = "1.25mean-by_ct"
        plt.subplot(2,1,2)
        plt.bar(dg.date_nl, dg.uv_ct, label="UV count")
        plt.plot(dg.date_nl, s_arr, label="Avg UV count", c="k")
    plt.legend()
    plt.grid()
    plt.subplot(2,1,1)
    if dgs.empty == False:
        plt.scatter(dgs.date_nl, dgs.RadE9_Mult_Nadir_Norm, s=6, c="b")
        plt.scatter(dgb.date_nl, dgb.RadE9_Mult_Nadir_Norm, s=6, c="r")
    else:
        plt.scatter(dgb.date_nl, dgb.RadE9_Mult_Nadir_Norm, s=6, c="r")
    plt.plot(dg.date_nl, mean_rad_arr, c="k", label="Average Rad")
    plt.plot(dg.date_nl, q1, '--', c="g", label="Q1")
    plt.plot(dg.date_nl, q3, '--', c="orange", label="Q2")
    plt.legend()
    plt.grid()
    plt.title("{}".format(name))
    plt.savefig("./outputs_esmi_temporal/{}/{}.png".format(folder,name))
    # plt.show()
    # code.interact(local = locals())
    return dgs, dgb

if __name__ == '__main__':
    comp_path = './nairobi_52/'
    dd, dc, min_esmi_dt, max_esmi_dt = esmi_data()
    dnl = daily_composites(comp_path, min_esmi_dt, max_esmi_dt)
    db = daily_voltage_events(dd,dc)
    db_combo = pd.merge(db, dnl, how='left', left_on=['id', 'ts_date'], right_on=['id', 'date_nl'])
    dcorr = corr_db(db_combo)
    db_combo2 = pd.merge(dnl, db, how="left", left_on=['id','date_nl'], right_on=['id', 'ts_date'])
    # raw_analysis("Mountain_View_02", db_combo2, dnl, type="duration")
    for i in db_combo2.id.unique():
        raw_analysis(i, db_combo2, dnl, type="duration")
    for i in db_combo2.id.unique():
        raw_analysis(i, db_combo2, dnl, type="ct")
    code.interact(local = locals())