import sys,time,re,statistics,xlrd,pyproj,datetime,pandas as pd,numpy as np,fiona,pickle,geojson,glob
from rasterstats import zonal_stats, point_query
from shapely.geometry import *
from shapely.ops import *
from osgeo import ogr,gdal
from gdalconst import *
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pandas as pd
import rasterio as rio
import code
import os
import matplotlib.pyplot as plt
import seaborn as sns

CURRENTDIR = os.getcwd()

def esmi_csv_to_geodf(path):
    df = pd.read_excel(path)
    geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
    df = df.drop(['Longitude', 'Latitude'], axis=1)
    crs = {'init': 'epsg:4326'} #same as the crs used by raster files
    gdf = GeoDataFrame(df, crs=crs, geometry=geometry)
    return gdf

def extract_point_values(gdf):
    #### We have two tiles of data out of the 6 tiles that NOAA provides.
    #### This can be checked using rio.open(tiff file).bounds
    #### Tile 2 ROI: 75N060W (File 0)
    #### Tile 5 ROI: 00N060W (File 1)
    #### This dataset has single band. Check by rio.open(tiff).count
    db0 = gdf.copy()
    db1 = gdf.copy()
    NL_dates = sorted(glob.glob('data/*'))
    for NL_date in NL_dates:
        # NL_files = glob.glob(NL_date+'/vcmcfg/*avg_rade9*.tif')

        #### Since the only tile which contains our coordinates is File 1 i.e.
        # 00N060W, we glob out only that file everytime in the loop and so
        # eventually we won't have to use stats0 and stats1 and we will
        # have just one variable.
        NL_files = glob.glob(NL_date+'/vcmcfg/*00N060W*'+'*avg_rade9h*.tif')
        stats0 = point_query(gdf, NL_files[0])
        db0[NL_date.split('/')[1]] = stats0

        # stats1 = point_query(gdf, NL_files[1])
        # db1[NL_date.split('/')[1]] = stats1

    # return db0, db1
    return db0

def prepare_analysis_image_db(db0, db1):
    # Extract list of columns of interest
    cols = db0.columns.values
    col_filt = ['Area', 'Income Level', 'Ward Name', 'geometry']
    col_filt += [x for x in cols if '20171' in x]
    col_filt += [x for x in cols if '20180' in x]
    col_filt += ['201810']
    # Filter dataframes using columns of interest list
    df0 = db0[col_filt]
    df1 = db1[col_filt]

    # Make a df1 as unified dataframe and then use it throughout
    #### will not be needed once we use the updated glob function
    #### updated glob will give one unified df.
    df1['201712'] = df0['201712']
    df1['201802'] = df0['201802']
    df1['201804'] = df0['201804']
    df1['201809'] = df0['201809']

    # melt the dataframe
    df = df1.copy()
    col_list = []
    col_list += [x for x in cols if '20171' in x]
    col_list += [x for x in cols if '20180' in x]
    col_list += ['201810']
    df = df.melt(id_vars = ['Area', 'Ward Name', 'Income Level', 'geometry'], value_vars = col_list, value_name = "pix_val")
    df = df.rename(columns = {'variable':'month_year'})
    df.month_year = pd.to_datetime(df.month_year, format="%Y%m")
    df['ts_month'] = df.month_year.apply(lambda x: x.month)
    df['ts_year'] = df.month_year.apply(lambda x: x.year)
    df = df.rename(columns = {'Ward Name':'Location_Name', 'Income Level':'Income_Level'})
    # code.interact(local = locals())
    return df

def trends_plotting_function(grouped_db, x, y1, y2, j):
    grouped = grouped_db
    rowlength = grouped.ngroups/10
    fig, axs = plt.subplots(figsize=(16,9),
                        nrows=11, ncols=int(rowlength),     # fix as above
                        gridspec_kw=dict(hspace=0.3)
                        )

    targets = zip(grouped.groups.keys(), axs.flatten())

    for i, (key, ax) in enumerate(targets):
        if j==1:
            print('yes 1')
            ax.plot(grouped.get_group(key)[x].apply(lambda x: x.strftime('%b, %Y')), grouped.get_group(key)[y1], marker = 'o')
        else:
            print('yes 2')
            ax.plot(grouped.get_group(key)[x], grouped.get_group(key)[[y1, y2]])

        ax.set_title('{}'.format(key))
        # ax.set_yticklabels([])
        ax.set_xticklabels([])

    # plt.tight_layout()
    plt.show()
    return None

def monthly_voltage_events(dd, dc):
    #create a table with site specific duration of different voltage event classes every month
    #we are considering only total duration of events over the month
    db_dur = dd.groupby(['Location_Name', 'v_class']).resample('1M', on='date_hour').count()[['date_hour']]
    db_dur = db_dur.rename(columns ={'date_hour':'event_dur'})
    db_dur.event_dur = db_dur.event_dur/60.0
    db_dur = db_dur.unstack(level = 1)
    db_dur.columns = db_dur.columns.droplevel(0)
    db_dur = db_dur.reset_index()

    #create a table with site specific count of different voltage event classes every month
    #we are considering only total actual count of events over the month
    db_ct = dc.groupby(['Location_Name', 'v_class']).resample('1M', on='min_date_hour').count()[['dur_hour']]
    db_ct = db_ct.unstack(level = 1)
    db_ct.columns = db_ct.columns.droplevel(0)
    db_ct = db_ct.reset_index()
    db_ct = db_ct.rename(columns = {'min_date_hour':'date_hour'})

    # merge both the dataframes
    db = pd.merge(db_dur, db_ct, how='left', left_on=['Location_Name', 'date_hour'], right_on = ['Location_Name', 'date_hour'])
    db = db.rename(columns = {'low_x':'low_dur', 'normal_x':'normal_dur', 'no_x':'no_dur', 'no_data_x':'no_data_dur', 'over_x':'over_dur', 'low_y':'low_ct', 'normal_y':'normal_ct', 'no_y':'no_ct', 'no_data_y':'no_data_ct', 'over_y':'over_ct'})
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
    # Lets say we remove all the points where image data is not available i.e. ts_month == 6
    # Also lets remove all the points where image pix value is 0 i.e. on month_year == '2018-04-01'
    db = db[(db.ts_month !=6) | (db.month_year != '2018-04-01')]
    d_corr = db.groupby(['Location_Name']).corr().reset_index()
    d_corr = d_corr[(d_corr.level_1 == 'pix_val')]
    # no data columns should not be used for correlation calculations
    d_corr.index = d_corr.Location_Name
    d_corr = d_corr.drop(columns = ['level_1', 'Location_Name', 'no_data_ct', 'no_data_dur', 'ts_month', 'ts_year', 'pix_val'])
    d_corr['max_col'] = abs(d_corr).idxmax(axis = 1)
    # code.interact(local = locals())
    return d_corr

if __name__ == '__main__':
    #---------------Monthly Variation Dataset Creation---------------------------------------

    #-----------Convert ESMI coordinates CSV to Shape Dataframe------------
    # gdf = esmi_csv_to_geodf('location_data.xlsx')

    #-----------Extracting point values------------------------------------
    # dg0, dg1 = extract_point_values(gdf.copy())
    # dg0.to_csv('tile0_stats.csv')
    # dg1.to_csv('tile1_stats.csv')

    #----------Monthly Variation Data analysis-----------------------------------------------
    dg0 = pd.read_csv(CURRENTDIR + '/cluster_output/tile0_stats.csv')
    dg1 = pd.read_csv(CURRENTDIR + '/cluster_output/tile1_stats.csv')

    df = prepare_analysis_image_db(dg0.copy(), dg1.copy())

    # Plotting data for complete ESMI date range
    # trends_plotting_function(df.groupby(['Location_Name']), 'month_year', 'pix_val', None, 1)

    # Plotting data after removing month of march since it has only zeros.
    # trends_plotting_function(df[(df.month_year.dt.month!=4)].groupby(['Location_Name']), 'month_year', 'pix_val', None, 1)

    #----------Unpickle ESMI Voltage Original and Events Dataset-----------------------------
    # We get total duration from original voltage database
    re = '/Users/zealshah/Documents/reliability_studies/ESMI_Kenya/'
    infile = open(re + 'esmi_kenya_column_db', 'rb')
    dd = pickle.load(infile)
    infile.close()
    dd = dd.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dd['v_class'] = dd.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")

    # We get event count from voltage events database
    infile = open('voltage_events_esmi_kenya', 'rb')
    dc = pickle.load(infile)
    infile.close()
    dc.event_date = pd.to_datetime(dc.event_date, format = "%Y-%m-%d")

    dv_total = monthly_voltage_events(dd.copy(), dc.copy())
    # dv_averages = daily_voltage_events(dd.copy(), dc.copy())

    #------------Combine Image data with Voltage data---------------------------------------
    # Left join on voltage data.
    # So we select image data corresponding to which the voltage data is available
    db_combo = pd.merge(dv_total, df, how='left', left_on=['Location_Name', 'ts_year', 'ts_month'], right_on = ['Location_Name', 'ts_year', 'ts_month'])
    dcorr = corr_db(db_combo.copy())
    code.interact(local = locals())