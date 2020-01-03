import pandas as pd
import numpy as np
import os
import statistics
import glob
import pickle
import matplotlib.pyplot as plt
import code

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
    print("Length of NL data before filtering: {}".format(len(dc)))

    # filter entries recorded before and after ESMI data date range
    dc = dc[(dc.Scan_Time<=max_esmi_dt) & (dc.Scan_Time>=min_esmi_dt)]
    print("Length of NL data after filtering: {}".format(len(dc)))

    # extracting only the columns we need
    # dc = dc[['id','Latitude','Longitude','Agg_Name','QF_Vflag','LI','Scan_Time','RadE9_DNB','RadE9_Mult_Nadir_Norm','Time_Segment']]
    return dc

def esmi_data(esmi_path):
    infile = open(esmi_path + 'esmi_kenya_column_db', 'rb')
    dd = pickle.load(infile)
    infile.close()
    dd = dd.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dd['v_class'] = dd.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")
    #convert Eastern African time to UTC
    dd['dt_utc'] = dd.date_hour.dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    #create columns for date, hours and minutes
    dd['date'] = dd.dt_utc.dt.date
    dd['hour'] = dd.dt_utc.dt.hour
    dd['minute'] = dd.dt_utc.dt.minute
    #adding id column to be used as key for NL data (basically just adds _ between name and number)
    dd['id'] = dd.Location_Name.apply(lambda x: str(x.split(' ')[0] + '_' + x.split(' ')[1]))
    return dd

def combined_db_inst(dc, de):
    ############################################################################
    #combining raw esmi and daily nl
    #NOTE: 1999 null values out of ~7500 which means no voltage reading during
    #the instance when scanning happened. So we filter out all the rows when
    #there was a scanned reading but no voltage reading
    ############################################################################
    #get a column of total scan days and rows representing every location
    dg0 = dc.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'scan_days'})
    #merge NL and voltage
    dg = pd.merge(dc, de, left_on=['id','date_nl','hour_nl','minute_nl'], right_on=['id','date','hour','minute'], how='left')
    #get column of lost days due to missing voltage data for every location
    dg1 = dg[(dg.voltage.isnull())]
    dg1 = dg1.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'lost_days'})
    # we only select the rows with both scan readings and voltage readings
    dg = dg[~(dg.voltage.isnull())]
    # also get a column of remaining days i.e. days with voltage data for every location
    dg2 = dg.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'remaining_days'})
    # create a summary table
    ds = dg0.merge(dg1,on='id',how='outer').merge(dg2,on='id',how='outer').fillna(0)
    print("Summary of data points left after merging NL and ESMI")
    print(ds)
    # remove all the unnecessary columns from combined db and send as output
    return dg[['id','Latitude','Longitude','Scan_Time','LI','RadE9_Mult_Nadir_Norm','voltage']]

def combined_db_hourly(dc, de):
    ############################################################################
    #combining raw esmi and daily nl
    #NOTE: 1999 null values out of ~7500 which means no voltage reading during
    #the instance when scanning happened. So we filter out all the rows when
    #there was a scanned reading but no voltage reading
    ############################################################################
    de = de.groupby(['id']).resample('1H', on='dt_utc').mean()[['voltage']].reset_index()
    de['date'] = de.dt_utc.dt.date
    de['hour'] = de.dt_utc.dt.hour
    #get a column of total scan days and rows representing every location
    dg0 = dc.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'scan_days'})
    #merge NL and voltage
    dg = pd.merge(dc, de, left_on=['id','date_nl','hour_nl'], right_on=['id','date','hour'], how='left')
    #get column of lost days due to missin voltage data for every location
    dg1 = dg[(dg.voltage.isnull())]
    dg1 = dg1.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'lost_days'})
    # we only select the rows with both scan readings and voltage readings
    dg = dg[~(dg.voltage.isnull())]
    # also get a column of remaining days i.e. days with voltage data for every location
    dg2 = dg.groupby(['id']).count()[['Scan_Time']].reset_index().rename(columns={'Scan_Time':'remaining_days'})
    # create a summary table
    ds = dg0.merge(dg1,on='id',how='outer').merge(dg2,on='id',how='outer').fillna(0)
    print("Summary of data points left after merging NL and ESMI")
    print(ds)
    # remove all the unnecessary columns from combined db and send as output
    return dg[['id','Latitude','Longitude','Scan_Time','LI','RadE9_Mult_Nadir_Norm','voltage']]

def corr_db(dg):
    dg1 = dg[['id','Scan_Time','RadE9_Mult_Nadir_Norm','voltage']]
    dr = dg1.groupby(['id']).corr().reset_index()
    dr = dr[(dr.level_1=="RadE9_Mult_Nadir_Norm")]
    dr = dr.drop(columns = ["RadE9_Mult_Nadir_Norm"])
    return dr

def trends_plotting_function(grouped_db, x, y1, y2, j):
    grouped = grouped_db
    rowlength = grouped.ngroups/10
    fig, axs = plt.subplots(figsize=(16,9),
                        nrows=11, ncols=int(rowlength),     # fix as above
                        gridspec_kw=dict(hspace=1.6)
                        )
    targets = zip(grouped.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        if j==1:
            print('yes 1')
            ax.scatter(grouped.get_group(key)[x], grouped.get_group(key)[y1], marker = 'o', s=2)
        else:
            print('yes 2')
            ax.scatter(grouped.get_group(key)[x], grouped.get_group(key)[[y1, y2]])
        ax.set_title('{}'.format(key))
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
    # plt.tight_layout()
    plt.show()
    return None

def plot_trends(dg):
    for i in dg.id.unique():
        dg1 = dg[(dg.id == i)]
        dg1 = dg1.sort_values(by=['voltage'])
        plt.scatter(dg1.voltage, dg1.RadE9_Mult_Nadir_Norm, s=2)
        plt.title("{}".format(i))
        plt.savefig("./nairobi_52_plots/{}.png".format(i))
        del(dg1)
    return None

if __name__ == '__main__':
    esmi_path = '/Users/zealshah/Documents/reliability_studies/ESMI_Kenya/'
    comp_path = './nairobi_52/'
    de = esmi_data(esmi_path)
    dc = daily_composites(comp_path, de.dt_utc.min().tz_convert(None), de.dt_utc.max().tz_convert(None))
    dg = combined_db_inst(dc.copy(),de.copy())
    dg_h = combined_db_hourly(dc.copy(),de.copy())
    dr_inst = corr_db(dg.copy())
    dr_h = corr_db(dg_h.copy())
    trends_plotting_function(dg.groupby(['id']), 'voltage', 'RadE9_Mult_Nadir_Norm', None, 1)
    # plot_relations(dg.copy())
    code.interact(local=locals())

