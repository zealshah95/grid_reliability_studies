import pandas as pd
import numpy as np
import os
import code
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

CURRENTDIR = os.getcwd()
re = CURRENTDIR + '/ESMI_Kenya/'

# dv = pd.read_excel(re + 'voltage_data_1.xlsx')

def create_sqlitedb():
    filename = "voltage_data_1"
    # filename = "interruptions_summary"
    con = sqlite3.connect(filename+".db")
    wb = pd.read_excel(re + filename + '.xlsx', sheet_name=None)
    for sheet in wb:
        wb[sheet].to_sql(sheet, con, index=False)
    con.commit()
    con.close()

def fetch_data():
    def area_name_change(df):
        a = df['Area']
        l = df['Location_Name']
        if l == "Mwiki 01" or l == "Mwiki 02" or l == "Mwiki 03" or l == "Mwiki 04" or l == "Mwiki 05" or l == "Mwiki 06":
            return "Mwiki"
        else:
            return df['Area']

    def income_name_change(df):
        l = df['Location_Name']
        i = df['Income_Level']
        if l == "Mwiki 01" or l == "Mwiki 02" or l == "Mwiki 03" or l == "Mwiki 04" or l == "Mwiki 05" or l == "Mwiki 06":
            return "Lower Middle Income"
        else:
            return df['Income_Level']

    voltage_db = re + 'voltage_data_1.db'
    interrupts_db = re + 'interruptions_summary.db'
    conn1 = sqlite3.connect(voltage_db)
    conn2 = sqlite3.connect(interrupts_db)
    dv = pd.read_sql_query("SELECT * FROM Sheet1;", conn1)
    conn1.close()
    di = pd.read_sql_query("SELECT * FROM Sheet1;", conn2)
    conn2.close()

    # # extract the list of locations and corresponding areas
    # # adds an "Area" column to interrupts dataset
    # d_a_loc = pd.DataFrame()
    # d_a_loc = dv[['Location_Name', 'Area']].drop_duplicates()
    # di = pd.merge(di, d_a_loc, on='Location_Name')

    # create a timestamp yy-mm-dd hh:00:00 using "hour" and "date"
    # splits the weird timestamp in date column
    dv.Date = dv.Date.apply(lambda x: x.split(" ", 1)[0])
    # converts hour to string, and prefixes a 0 to single digit
    # numbers for correct datetime formatting
    dv.Hour_of_day = dv.Hour_of_day.astype(str)
    dv.Hour_of_day = dv.Hour_of_day.apply(lambda x: x.zfill(2))
    # combines date and hour with space in between
    dv['date_hour'] = dv.Date +' '+ dv.Hour_of_day
    # converts string date time to Timestamp
    dv.date_hour = pd.to_datetime(dv.date_hour, format="%Y-%m-%d %H")
    # Area named "Mountain" was replaced by "Mountain View" after 3 days of deployment
    # for consistency we replace Mountain with Mountain View
    dv['Area'] = np.where(dv['Area']== "Mountain", "Mountain View", dv['Area'])
    # For some timestamp values, the Area name for ‘Mwiki 02’, ‘Mwiki 01’, ‘Mwiki 03’,
    # ‘Mwiki 04’, ‘Mwiki 06’ was changed to Kawangware. So we change it back to "Mwiki"
    dv['Area'] = dv.apply(area_name_change, axis = 1)
    dv['Income_Level'] = dv.apply(income_name_change, axis = 1)
    dv = dv.drop_duplicates()

    ##### RESAMPLING CODE GOT DELETED
    # maybe use dv[dv.isnull()]==None for ease of work...???
    return dv, di

def availability_analysis(da):
    def daily_av_tiers(df):
        d = df['availability']
        e = df['eve_availability']
        if 0<=d<4:
            return 0
        # tie breaker for daily availability value of minimum 4 i.e. tier 1 or 2
        elif 4<=d<8 and 1<=e<2:
            return 1
        elif 4<=d<8 and 2<=e:
            return 2
        elif 8<=d<16:
            return 3
        elif 16<=d<23:
            return 4
        elif 23<=d<=24:
            return 5

    def eve_av_tiers(df):
        d = df['availability']
        e = df['eve_availability']
        if (0<=e<1):
            return 0
        elif (1<=e<2):
            return 1
        elif (2<=e<3):
            return 2
        elif (3<=e<4):
            return 3
        # tie breaker for eve availability tier 4 and 5
        elif (4<=e) & (16<=d<23):
            return 4
        elif (4<=e) & (23<=d<=24):
            return 5
        elif (4<=e):
            return 5

    cols = da.columns.values
    # extract Min_01 to Min_60 column names
    col_list = [x for x in cols if 'Min' in x]
    # Replace all voltage values with V>100V as 1
    # This is possibly replacing Nans with zeros...!!!
    # da[col_list] = np.where(da[col_list]>100, 1, da[col_list])
    # da[col_list] = np.where((da[col_list] == 1) | (da[col_list] == None), da[col_list], 0)

    # No data is when the entries are Nan
    # da['no_data'] = 60 - da[col_list].count(axis = 1) #good alternative
    da['no_data'] = da[col_list].isnull().sum(axis = 1)

    # total availability in minutes during every hour of the day
    # V>100V is considered available
    # This dataset is used as a parent database for availability
    # resampling on daily and evening basis
    da['availability'] = da[(da[col_list] > 100)].count(axis = 1)

    #-------------daily availability dataset-------------------------
    daily_av = da.groupby(['Area', 'Location_Name']).resample('1D', on='date_hour').sum()[['availability', 'no_data']].reset_index()
    daily_av[['availability', 'no_data']]= daily_av[['availability', 'no_data']].apply(lambda x: x/60.0) #hours

    # Filter our portions with no data
    # if no_data is less than 24 hours, it means there was atleast one 1 minute interval in that day
    # when the data was recorded
    daily_av = daily_av[(daily_av.no_data < 24)]
    daily_av = daily_av.drop(columns = ['no_data'])

    #-------------evening availability dataset-----------------------
    # evening ~ 6 hours after sunset i.e. 6pm to 11pm in Nairobi
    eve_av = da[(da.date_hour.dt.hour >= 18) & (da.date_hour.dt.hour <= 23)].groupby(['Area','Location_Name']).resample('1D', on='date_hour').sum()[['availability', 'no_data']].reset_index()
    eve_av.rename(columns = {'availability':'eve_availability'}, inplace=True)
    eve_av[['eve_availability', 'no_data']] = eve_av[['eve_availability', 'no_data']].apply(lambda x: x/60.0) #hours
    # Filter our portions with no data
    # if no_data is less than 4 hours, it means there was atleast one 1 minute interval during that evening
    # when the data was recorded
    eve_av = eve_av[(eve_av.no_data < 4)]
    eve_av = eve_av.drop(columns = ['no_data'])

    #------combining daily and eve availability datasets-------------
    avail_db = pd.merge(daily_av, eve_av, how='outer', on=['Area', 'Location_Name', 'date_hour'])
    # calculate tiers for availability indices
    avail_db['daily_availability_tier'] = avail_db.apply(daily_av_tiers, axis = 1)
    avail_db['eve_availability_tier'] = avail_db.apply(eve_av_tiers, axis = 1)

    return avail_db

def reliability_analysis(dr):
    #### STICK WITH ESMI DEFINITIONS
    #### DEFINE INTERRUPTIONS/DISRUPTIONS OF SUPPLY: < 100V will be a disruption
    #### DEFINE SAIDI SAIFI
    def disruption_tiers(df):
        n = df['disrup_count']
        du = df['disrup_duration']
        # SOURCE: WB ETHIOPIA document page 95
        if n<=3 and du<=2:
            return 5
        elif n<=3 and du>2:
            return 4
        elif 3<=n<=14:
            return 4
        else:
            return np.nan

    def saifi_saidi_tiers(df):
        f = df['SAIFI']
        d = df['SAIDI']
        if f<156 and d<6240:
            return 5
        elif f<730:
            return 4
        else:
            return np.nan

    def pq_tier_assign(df):
        pq_c = df['pq_count']
        if pq_c>0:
            return np.nan
        else:
            return 5

    cols = dr.columns.values
    # extract Min_01 to Min_60 column names
    col_list = [x for x in cols if 'Min' in x]

    # No data is when the entries are Nan
    # da['no_data'] = 60 - da[col_list].count(axis = 1) #good alternative
    dr['no_data'] = dr[col_list].isnull().sum(axis = 1)

    # V<100V is considered a disruption
    dr['disrup_duration'] = dr[(dr[col_list] < 100)].count(axis = 1) #minutes
    # if there were one or more disruptions in an hour, we consider it as 1 disruption count
    # We are counting only long duration outages i.e. >= 1hour duration.
    dr['disrup_count'] = dr[(dr[col_list] < 100)].count(axis = 1)
    dr['disrup_count'] = np.where(dr['disrup_count']>0, 1, dr['disrup_count'])
    #### date ranges for which data is available for a specific location
    # ds['data_duration'] = dr.groupby(['Area', 'Location_Name']).apply()

    #-------------weekly reliability dataset-------------------------
    reliab_db = dr.groupby(['Area', 'Location_Name']).resample('1W', on='date_hour').sum()[['disrup_count', 'disrup_duration', 'no_data']].reset_index()
    reliab_db[['disrup_duration', 'no_data']] = reliab_db[['disrup_duration', 'no_data']].apply(lambda x: x/60.0) #hours
    reliab_db['disrup_tier'] = reliab_db.apply(disruption_tiers, axis = 1)

    #--------------------SAIDI and SAIFI-----------------------------
    # duration is in minutes for saidi saifi dataset
    # we have exactly 1 year of data so we directly sum the disruption counts and duration values
    reliab_db2 = dr.groupby(['Area', 'Location_Name']).sum()[['disrup_count', 'disrup_duration']].reset_index()
    reliab_db2 = reliab_db2.groupby(['Area']).aggregate({'Location_Name':['nunique'],'disrup_count':'sum','disrup_duration':'sum'}).reset_index()
    reliab_db2.columns = reliab_db2.columns.droplevel(1)
    reliab_db2['SAIFI'] = reliab_db2['disrup_count']/reliab_db2['Location_Name']
    reliab_db2['SAIDI'] = reliab_db2['disrup_duration']/reliab_db2['Location_Name']
    reliab_db2['reliab_index_tiers'] = reliab_db2.apply(saifi_saidi_tiers, axis = 1)

    return reliab_db, reliab_db2

def pq_analysis(dm):
    # We are considering only hourly voltage problems
    dpq = dm.copy()
    cols = dpq.columns.values
    # extract Min_01 to Min_60 column names
    col_list = [x for x in cols if 'Min' in x]
    #--------------number of long duration and short duration outages-------------------------
    # dpq['total_outages'] = dpq[(dpq[col_list] < 100)].count(axis = 1)
    # dpq['long_duration_outages'] = np.where(dpq['total_outages'] == 60, 1, 0)
    # dpq['short_duration_outages'] = np.where(dpq['total_outages'] < 60, 1, 0)

    #-----percentage of time for which voltage was low, normal and high------------------------
    dpq['no_supply_duration'] = dpq[(dpq[col_list] < 101)].count(axis = 1)
    dpq['low_voltage_duration'] = dpq[(dpq[col_list] >= 101) & (dpq[col_list] < 226)].count(axis = 1)
    dpq['normal_voltage_duration'] = dpq[(dpq[col_list] >= 226) & (dpq[col_list] <254)].count(axis = 1)
    dpq['over_voltage_duration'] = dpq[(dpq[col_list] >= 254)].count(axis = 1)
    dpq['total_dur'] = dpq['no_supply_duration'] + dpq['low_voltage_duration'] + dpq['normal_voltage_duration'] + dpq['over_voltage_duration']
    # resample by hour to get all the non-recorded timestamps and create no data duration column
    dpq = dpq.groupby(['Area', 'Location_Name']).resample('1H', on='date_hour').mean().reset_index()
    dpq['no_data'] = dpq[col_list].isnull().sum(axis = 1)

    code.interact(local = locals())

    # DAILY PQ issues duration in hours
    dpq = dpq.groupby(['Area', 'Location_Name']).resample('1D', on = 'date_hour').sum()[['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'total_dur', 'no_data']].reset_index()
    dpq[['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'total_dur', 'no_data']] = dpq[['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'total_dur', 'no_data']].apply(lambda x: x/60.0)
    # sites with incomplete date range causing uneven total duration and no data values need to be filtered
    # we remove all the points where sum of total duration and no_data is not equal to 24
    dpq['to_be_filtered'] = dpq['total_dur'] + dpq['no_data']
    dpq = dpq[(dpq.to_be_filtered == 24.0)]
    return dpq

def customer_classification(db, df_av, df_rel, df_ind, df_pq):
    d_i = pd.DataFrame()
    d_i = db[['Location_Name', 'Income_Level']].drop_duplicates()

    #------------daily and eve availability------------------------------------------
    # Add income level details to the dataset
    df_av = df_av.groupby(['Area', 'Location_Name']).mean().reset_index()
    df_av = pd.merge(df_av, d_i, on='Location_Name')

    #------daily availability vs. income---------------
    # create a dataset with voltage ranges as columns, income values as rows and number
    # of households as cell values
    av_plot = pd.DataFrame(index = df_av.Income_Level.unique())
    av_plot['less_than_4'] = df_av[(df_av.availability<4)].groupby(['Income_Level']).Location_Name.nunique()
    av_plot['4-8'] = df_av[(df_av.availability<8) & (df_av.availability>=4)].groupby(['Income_Level']).Location_Name.nunique()
    av_plot['8-16'] = df_av[(df_av.availability<16) & (df_av.availability>=8)].groupby(['Income_Level']).Location_Name.nunique()
    av_plot['16-23'] = df_av[(df_av.availability<23) & (df_av.availability>=16)].groupby(['Income_Level']).Location_Name.nunique()
    av_plot['23_or_more'] = df_av[(df_av.availability>=23)].groupby(['Income_Level']).Location_Name.nunique()
    av_plot['total_customers'] = av_plot.sum(axis = 1)
    # add an overall summary division as well
    df_av_append = av_plot.sum(axis = 0)
    df_av_append.name = "overall"
    av_plot = av_plot.append(df_av_append)

    #------evening availability vs. income-------------
    ev_av_plot = pd.DataFrame(index = df_av.Income_Level.unique())
    ev_av_plot['less_than_1'] = df_av[(df_av.eve_availability<1)].groupby(['Income_Level']).Location_Name.nunique()
    ev_av_plot['1-2'] = df_av[(df_av.eve_availability<2) & (df_av.eve_availability>=1)].groupby(['Income_Level']).Location_Name.nunique()
    ev_av_plot['2-3'] = df_av[(df_av.eve_availability<3) & (df_av.eve_availability>=2)].groupby(['Income_Level']).Location_Name.nunique()
    ev_av_plot['3-4'] = df_av[(df_av.eve_availability<4) & (df_av.eve_availability>=3)].groupby(['Income_Level']).Location_Name.nunique()
    ev_av_plot['4_or_more'] = df_av[(df_av.eve_availability>=4)].groupby(['Income_Level']).Location_Name.nunique()
    ev_av_plot['total_customers'] = ev_av_plot.sum(axis = 1)
    # add an overall summary division as well
    df_ev_av_append = ev_av_plot.sum(axis = 0)
    df_ev_av_append.name = "overall"
    ev_av_plot = ev_av_plot.append(df_ev_av_append)

    #------------reliability------------------------------------------------------------
    # weekly outages count for every customer
    df_rel = df_rel.groupby(['Area', 'Location_Name']).mean().reset_index()
    df_rel = pd.merge(df_rel, d_i, on='Location_Name')

    #------------weekly outage count vs. income---------------
    r_ct_plot = pd.DataFrame(index = df_rel.Income_Level.unique())
    r_ct_plot['less_than_4'] = df_rel[(df_rel.disrup_count < 4)].groupby(['Income_Level']).Location_Name.nunique()
    r_ct_plot['4-14'] = df_rel[(df_rel.disrup_count >= 4) & (df_rel.disrup_count < 14)].groupby(['Income_Level']).Location_Name.nunique()
    r_ct_plot['14-30'] = df_rel[(df_rel.disrup_count >= 14) & (df_rel.disrup_count < 30)].groupby(['Income_Level']).Location_Name.nunique()
    r_ct_plot['30_or_more'] = df_rel[(df_rel.disrup_count >= 30)].groupby(['Income_Level']).Location_Name.nunique()
    r_ct_plot['total_customers'] = r_ct_plot.sum(axis = 1)

    df_rel_ct_append = r_ct_plot.sum(axis = 0)
    df_rel_ct_append.name = "overall"
    r_ct_plot = r_ct_plot.append(df_rel_ct_append)

    #-----------weekly outage duration vs. income-------------
    r_d_plot = pd.DataFrame(index = df_rel.Income_Level.unique())
    r_d_plot['less_than_2'] = df_rel[(df_rel.disrup_duration < 2)].groupby(['Income_Level']).Location_Name.nunique()
    r_d_plot['2-5'] = df_rel[(df_rel.disrup_duration >= 2) & (df_rel.disrup_duration < 5)].groupby(['Income_Level']).Location_Name.nunique()
    r_d_plot['5-12'] = df_rel[(df_rel.disrup_duration >= 5) & (df_rel.disrup_duration < 12)].groupby(['Income_Level']).Location_Name.nunique()
    r_d_plot['12_or_more'] = df_rel[(df_rel.disrup_duration >= 12)].groupby(['Income_Level']).Location_Name.nunique()
    r_d_plot['total_customers'] = r_d_plot.sum(axis = 1)

    df_rel_d_append = r_d_plot.sum(axis = 0)
    df_rel_d_append.name = "overall"
    r_d_plot = r_d_plot.append(df_rel_d_append)

    ##--------------saifi-saidi----------------------------------------------------------
    #### Need to make it relative since not all the sites have data for same time range...!!!!!!!
    #### Do not use much for now
    # area-wise income levels
    d_i_a = pd.DataFrame()
    d_i_a = db[['Area', 'Income_Level']].drop_duplicates()
    df_ind = pd.merge(df_ind, d_i_a, on = 'Area')
    # area and income-wise saidi-saifi
    df_ind = df_ind.sort_values(by = ['Income_Level'])

    ##---------------daily power quality-------------------------------------------------------
    # Average daily PQ status duration over one year
    df_pq = df_pq.groupby(['Area', 'Location_Name']).mean().reset_index()
    df_pq = pd.merge(df_pq, d_i, on = 'Location_Name')

    #-------mean pq duration faced by each area-income group--------------
    pq_plot = df_pq.groupby(['Area', 'Income_Level']).mean().reset_index()
    # percentage duration
    pq_plot[['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'no_data']].apply(lambda x: (x*100.0)/24.0)
    # code.interact(local=locals())

    return av_plot, ev_av_plot, r_ct_plot, r_d_plot, df_ind, pq_plot

def customer_classification_visualization(dm, av, ev_av, r_ct, r_d, db_ind, pq):
    #----------------------PRELIMINARY DATA EXPLORATION------------------------------
    dg = dm.copy()

    d1 = dg.aggregate({"Area":['nunique'],"Location_Name":['nunique'], "Income_Level":['nunique']})
    tab_1 = pd.DataFrame({'Total No. of Areas': d1.Area, 'Total No. of Sites':d1.Location_Name, 'Total No. of Income Groups':d1.Income_Level})

    d2 = dg.groupby(['Area']).nunique().Location_Name.reset_index()
    d2 = d2.rename(columns = {'Location_Name': 'Total No. of Sites'})

    d3 = dg.groupby(['Income_Level']).nunique().Location_Name.reset_index()
    d3 = d3.rename(columns = {'Location_Name': 'Total No. of Sites'})

    d4 = dg.groupby(['Location_Name']).apply(lambda x: x.date_hour.max() - x.date_hour.min()).reset_index()
    d4 = d4.rename(columns = {0:'Data Availability Duration (days)'})
    d4['Data Availability Duration (days)'] = d4['Data Availability Duration (days)'].apply(lambda x: x.days)

    #----------DAILY AVAILABILITY----------------------------------------------------
    #study variation in availability for every income group
    f_avail = plt.figure(1)
    av_plot_avail = av.apply(lambda x: x*100.0/x['total_customers'], axis = 1)
    av_plot_avail = av_plot_avail.drop(columns = ['total_customers'], axis = 1)
    #send for plotting using
    g_avail = av_plot_avail.plot(kind = 'bar', stacked = True)
    g_avail.set_xticklabels(g_avail.get_xticklabels(), rotation = 0)
    g_avail.set_title("Variation in Daily Supply Availability by Income")

    #study variation in income for every availability group
    f_income = plt.figure(2)
    av_plot_income = av.apply(lambda x: x*100.0/x['overall'])
    av_plot_income = av_plot_income.drop(['overall'])
    # send for plotting using
    g_av_income = av_plot_income.T.plot(kind = 'bar', stacked = True)
    g_av_income.set_xticklabels(g_av_income.get_xticklabels(), rotation = 0)
    g_av_income.set_title("Distribution of Income Groups by Daily Supply Availability")

    #----------EVE AVAILABILITY------------------------------------------------------
    # study variation in evening availability for every income group
    f_avail_ev = plt.figure(3)
    ev_av_plot_avail = ev_av.apply(lambda x: x*100.0/x['total_customers'], axis = 1)
    ev_av_plot_avail = ev_av_plot_avail.drop(columns = ['total_customers'], axis = 1)
    #send for plotting using
    g_avail_ev = ev_av_plot_avail.plot(kind = 'bar', stacked = True)
    g_avail_ev.set_xticklabels(g_avail_ev.get_xticklabels(), rotation = 0)
    g_avail_ev.set_title("Variation in Evening Supply Availability by Income")

    #study variation in income for every evening availability group
    f_income_ev = plt.figure(4)
    ev_av_plot_income = ev_av.apply(lambda x: x*100.0/x['overall'])
    ev_av_plot_income = ev_av_plot_income.drop(['overall'])
    # send for plotting using
    g_av_income_ev = ev_av_plot_income.T.plot(kind = 'bar', stacked = True)
    g_av_income_ev.set_xticklabels(g_av_income_ev.get_xticklabels(), rotation = 0)
    g_av_income_ev.set_title("Distribution of Income Groups by Evening Supply Availability")

    #-----------------RELIABILITY COUNT-----------------------------------------------
    # study variation in number of weekly outages for every income group
    f_r_ct = plt.figure(5)
    r_ct_plot = r_ct.apply(lambda x: x*100.0/x['total_customers'], axis = 1)
    r_ct_plot = r_ct_plot.drop(columns = ['total_customers'], axis = 1)
    # send for plotting using
    g_r_ct = r_ct_plot.plot(kind = 'bar', stacked = True)
    g_r_ct.set_xticklabels(g_r_ct.get_xticklabels(), rotation = 0)
    g_r_ct.set_title("Variation in Weekly Outage Count by Income")

    # study variation in income distribution for every outage category
    f_r_ct_income = plt.figure(6)
    r_ct_plot_income = r_ct.apply(lambda x: x*100.0/x["overall"])
    r_ct_plot_income = r_ct_plot_income.drop(['overall'])
    # send for plotting using
    g_ct_income = r_ct_plot_income.T.plot(kind = 'bar', stacked = True)
    g_ct_income.set_xticklabels(g_ct_income.get_xticklabels(), rotation = 0)
    g_ct_income.set_title("Distribution of Income Groups by Weekly Outage Count")

    #---------------RELIABILITY DURATION----------------------------------------------
    # study variation in duration of outages for every income group
    f_r_d = plt.figure(7)
    r_d_plot = r_d.apply(lambda x: x*100.0/x['total_customers'], axis = 1)
    r_d_plot = r_d_plot.drop(columns = ['total_customers'], axis = 1)
    # send for plotting using
    g_r_d = r_d_plot.plot(kind = 'bar', stacked = True)
    g_r_d.set_xticklabels(g_r_ct.get_xticklabels(), rotation = 0)
    g_r_d.set_title("Variation in Weekly Outage Duration(hours) by Income")

    # study variation in income distribution for every outage category
    f_r_d_income = plt.figure(8)
    r_d_plot_income = r_d.apply(lambda x: x*100.0/x["overall"])
    r_d_plot_income = r_d_plot_income.drop(['overall'])
    # send for plotting using
    g_d_income = r_d_plot_income.T.plot(kind = 'bar', stacked = True)
    g_d_income.set_xticklabels(g_d_income.get_xticklabels(), rotation = 0)
    g_d_income.set_title("Distribution of Income Groups by Weekly Outage Duration(hours)")

    # #-----------------------SAIFI-SAIDI----------------------------------------------
    # # convert SAIDI from minutes per customer to hours per customer
    # db_ind.SAIDI = db_ind.SAIDI.apply(lambda x: x/60.0)
    # # send for plotting using
    # f_saifi = plt.figure(9)
    # g_saifi = sns.barplot(x = db_ind.SAIFI, y = db_ind.Area).set_title("SAIFI per Area")

    # f_saidi = plt.figure(10)
    # g_saidi = sns.barplot(x = db_ind.SAIDI, y = db_ind.Area).set_title("SAIDI per Area")

    #-----------------------Power Quality----------------------------------------------
    cols = ['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'no_data']
    # % Average daily power quality duration by area
    pq_area = pq[['Area','no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'no_data']]
    pq_area[cols] = pq_area[cols].apply(lambda x: x*100.0/24.0)
    pq_area.index = pq_area.Area
    pq_area = pq_area.drop(columns = ['Area'])
    g_pq_area = pq_area.plot.barh(stacked = True).set_title("Average Daily Voltage Quality by Area")

    # % Average daily power quality duration by income
    pq_income = pq.groupby(['Income_Level']).mean()[cols].reset_index()
    pq_income[cols] = pq_income[cols].apply(lambda x: x*100.0/24.0)
    pq_income.index = pq_income.Income_Level
    pq_income = pq_income.drop(columns = ['Income_Level'])
    g_pq_income = pq_income.plot.barh(stacked = True).set_title("Average Daily Voltage Quality by Income")

    plt.show()
    # code.interact(local = locals())
    return None

def trends_plotting_function(grouped_db, x, y1, y2, j):
    grouped = grouped_db
    rowlength = grouped.ngroups/10
    fig, axs = plt.subplots(figsize=(9,4),
                        nrows=11, ncols=int(rowlength),     # fix as above
                        gridspec_kw=dict(hspace=0.4))
    targets = zip(grouped.groups.keys(), axs.flatten())

    for i, (key, ax) in enumerate(targets):
        if j==1:
            print('yes 1')
            ax.plot(grouped.get_group(key)[x], grouped.get_group(key)[y1])
        else:
            print('yes 2')
            ax.plot(grouped.get_group(key)[x], grouped.get_group(key)[[y1, y2]])

        ax.set_title('{}'.format(key))
        # ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()
    return None

def trends_study(av, r, r_ind, pq):
    # daily and evening availability
    # weekly outage count and duration
    # saidi saifi that we are not using for now
    # daily power quality stats
    #---------------------------------------PQ Trends------------------------------------------------
    cols = ['no_supply_duration', 'low_voltage_duration', 'normal_voltage_duration', 'over_voltage_duration', 'no_data']
    pq = pq.copy()
    pq = pq.groupby(['Area', 'Location_Name']).resample('1M', on='date_hour').mean()[cols].reset_index()
    grouped_pq = pq.groupby(['Location_Name'])
    #-------Monthly average daily low voltage duration---------------
    trends_plotting_function(grouped_pq, 'date_hour', 'low_voltage_duration', None, 1)
    #-------Monthly average daily low voltage duration---------------
    trends_plotting_function(grouped_pq, 'date_hour', 'no_supply_duration', None, 1)

    #----------------------------------Supply Availability Trends--------------------------------------
    av = av.copy()
    av = av.groupby(['Area', 'Location_Name']).resample('1M', on='date_hour').mean()[['availability', 'eve_availability']].reset_index()
    grouped_av = av.groupby(['Location_Name'])
    #-------Monthly average daily supply availability----------------
    trends_plotting_function(grouped_av, 'date_hour', 'availability', 'eve_availability', 2)

    # code.interact(local = locals())
    return None

def voltage_events(dm):
    # We are considering only hourly voltage problems
    dve = dm.copy()
    cols = dve.columns.values
    # extract Min_01 to Min_60 column names
    col_list = [x for x in cols if 'Min' in x]
    # # resampling to introduce missing data timestamps
    # dve = dve.groupby(['Area', 'Location_Name']).resample('1H', on='date_hour').mean().reset_index()
    # melting the minute columns to rows
    df = dve.melt(id_vars = ['Area', 'Location_Name', 'date'], value_vars = col_list, value_name = "voltage")
    # incorporating minutes to date_hour
    df = df.rename(columns = {'variable':'minute'})
    df.minute = df.minute.apply(lambda x: x.split("_")[1])
    # minutes go from 1 to 60, so lets convert it to 0 to 59 and prefix zeroes to it
    df.minute = df.minute.apply(lambda x: int(x)-1)
    df.minute = df.minute.apply(lambda x: str(x).zfill(2))

    # df.date_hour = df.date_hour.apply(lambda x: x.dt.minute == df.minute)
    # df['pq_duration'] = df.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over")
    code.interact(local = locals())

if __name__ == '__main__':
    # create_sqlitedb()
    volt_db, int_db = fetch_data()
    # # code.interact(local=locals())
    # # dataset for daily and evening availability analysis
    # df1 = availability_analysis(volt_db.copy())
    # # dataset for weekly disruptions or reliability analysis
    # # dataset for annual supply system outage analysis
    # df2, df3 = reliability_analysis(volt_db.copy())
    # # power quality duration
    # df4 = pq_analysis(volt_db.copy())
    voltage_events(dh_volt.copy())
    # df_av, df_ev_av, df_r_ct, df_r_d, df_saifi_saidi, df_pq = customer_classification(volt_db.copy(), df1.copy(), df2.copy(), df3.copy(), df4.copy())
    # customer_classification_visualization(volt_db.copy(), df_av.copy(), df_ev_av.copy(), df_r_ct.copy(), df_r_d.copy(), df_saifi_saidi.copy(), df_pq.copy())
    # trends_study(df1.copy(), df2.copy(), df3.copy(), df4.copy())
    # code.interact(local=locals())