"""
This code can be used to create FBBU area hulls using transformer locations.

Output:
fbbu_to_nl - fbbu name to nl file association dictionary
nl_to_fbbu - nl file to fbbu dictionary
"""

import shapely
from shapely.geometry import *
import pandas as pd
import geopandas as gpd
import numpy as np
import code
import datetime
import pickle
from rtree import index

def create_tx_fbbu_dataframe():
    #############################################################################
    # This code creates a data frame containing FBBU Names and corresponding
    # transformer codes and locations
    #############################################################################
    #Read outage data for 2017-18
    df_17 = pd.read_excel('../incidence_data/incidences2017.xlsx')
    df_18 = pd.read_excel('../incidence_data/incidences2018.xlsx')
    df_ims = pd.concat([df_17, df_18])
    # Filter out all the entries not belonging to Nairobi
    # **********CONFIRM WITH JAY*************************
    df_ims = df_ims[(df_ims.COUNTY == "Nairobi West County") | (df_ims.COUNTY == "Nairobi North County") | (df_ims.COUNTY == "Nairobi South County")]

    #Read transformer locations
    dtx = pd.read_pickle("../incidence_data/TXdatafull4.pck")
    dtx = dtx[(dtx.county == "Nairobi")] #filter our all transformers not in Nairobi county

    #Combine TX and IMS data using Internal code values
    # Output: Tx locations associated with every FBBU_NAME
    dm = pd.merge(df_ims, dtx[['Internal code','LAT','LON']], left_on=['INSTALATION'], right_on=['Internal code'], how="left")
    dm = dm[(~dm["Internal code"].isnull())][['Internal code','INSTALATION', 'FBBU_NAME', 'LAT', 'LON']].drop_duplicates()

    return dm

def create_convex_hulls(df):
    #############################################################################
    # This code creates convex hulls for FBBU_NAME field based on transformer
    # locations associated with each FBBU_NAME
    #############################################################################
    geoms = [(k[1],k[0]) for k in zip(df.LAT, df.LON)]
    df['tx_locs'] = geoms
    df = df.groupby(['FBBU_NAME']).tx_locs.apply(lambda x: np.asarray(x)).reset_index()
    df['total_tx'] = df.tx_locs.apply(lambda x: len(x)) #total tx in that FBBU region
    df['geometry'] = df.tx_locs.apply(lambda x: MultiPoint(x).convex_hull) #geometry column contains hulls
    df['area'] = df.geometry.apply(lambda x: x.area)
    gdf = gpd.GeoDataFrame(df, geometry = df.geometry, crs={'init':'epsg:4326'})
    gdf = gdf.drop(columns = ['tx_locs'])
    return gdf

def fbbu_NLgrid_association(dh):
    dh = dh.copy()
    dnl = pd.read_excel("nairobi_grid_points.xlsx")
    # geometry = [Point(xy) for xy in zip(dnl.lon, dnl.lat)]
    # dnl = gpd.GeoDataFrame(dnl, geometry=geometry, crs={'init':'epsg:4326'})
    fbbu_nl_list = {} #list of grid cells corresponding to a given fbbu
    nl_fbbu_list = {} #fbbu containing the given nl grid cell
    cell_idx = index.Index()
    for NL_index in dnl.index:
        cell_idx.insert(NL_index,(dnl['lon'][NL_index],dnl['lat'][NL_index],dnl['lon'][NL_index],dnl['lat'][NL_index]))

    for i in range(len(dh)):
        fbbu_name = dh.iloc[i].FBBU_NAME
        fbbu_poly = dh.iloc[i].geometry
        all_hits = list(cell_idx.intersection(fbbu_poly.bounds))
        fbbu_nl_list[fbbu_name] = [dnl['id'][hit] for hit in all_hits if fbbu_poly.contains(Point(dnl['lon'][hit],dnl['lat'][hit]))]
        for hit in all_hits:
            if fbbu_poly.contains(Point(dnl['lon'][hit],dnl['lat'][hit])):
                nl_fbbu_list[dnl['id'][hit]] = fbbu_name

    # pickle.dump(fbbu_nl_list,open('fbbu_to_nl.pck','wb'))
    # pickle.dump(nl_fbbu_list,open('nl_to_fbbu.pck','wb'))
    # code.interact(local = locals())
    return fbbu_nl_list, nl_fbbu_list

if __name__ == '__main__':
    dd = create_tx_fbbu_dataframe()
    dh = create_convex_hulls(dd)
    fbbu, nl = fbbu_NLgrid_association(dh)
    code.interact(local = locals())
