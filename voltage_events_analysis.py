import pandas as pd
import numpy as np
import os
import code
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
from plotly import tools
import jinja2
from sklearn.cluster import KMeans
from matplotlib.patches import Arc
from scipy.spatial import ConvexHull
from collections import OrderedDict
from scipy.stats import gaussian_kde
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# import plotly.express as px


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
    print("Database extracted from SQL db")

    # Area named "Mountain" was replaced by "Mountain View" after 3 days of deployment
    # for consistency we replace Mountain with Mountain View
    dv['Area'] = np.where(dv['Area']== "Mountain", "Mountain View", dv['Area'])
    # For some timestamp values, the Area name for ‘Mwiki 02’, ‘Mwiki 01’, ‘Mwiki 03’,
    # ‘Mwiki 04’, ‘Mwiki 06’ was changed to Kawangware. So we change it back to "Mwiki"
    dv['Area'] = dv.apply(area_name_change, axis = 1)
    dv['Income_Level'] = dv.apply(income_name_change, axis = 1)

    print("Names of sites and areas updated")

    # extract Min_01 to Min_60 column names
    cols = dv.columns.values
    col_list = [x for x in cols if 'Min' in x]
    # create a timestamp yy-mm-dd hh:00:00 using "hour" and "date"
    # splits the weird timestamp in date column
    dv.Date = dv.Date.apply(lambda x: x.split(" ", 1)[0])
    # converts hour to string, and prefixes a 0 to single digit
    # numbers for correct datetime formatting
    dv.Hour_of_day = dv.Hour_of_day.astype(str)
    dv.Hour_of_day = dv.Hour_of_day.apply(lambda x: x.zfill(2))

    print("Date and Hour split done")

    # melting the minute columns to rows
    dv = dv.melt(id_vars = ['Area', 'Location_Name', 'Income_Level', 'Date', 'Hour_of_day'], value_vars = col_list, value_name = "voltage")

    print("Melting Done")
    # incorporating minutes to date_hour
    dv = dv.rename(columns = {'variable':'minute'})
    dv.minute = dv.minute.apply(lambda x: x.split("_")[1])
    # minutes go from 1 to 60, so lets convert it to 0 to 59 and prefix zeroes to it
    dv.minute = dv.minute.apply(lambda x: int(x)-1)
    dv.minute = dv.minute.apply(lambda x: str(x).zfill(2))

    print("Minute level manipulation done")

    # combines date and hour with space in between
    dv['date_hour'] = dv.Date +' '+ dv.Hour_of_day +' '+dv.minute

    print("date_hour created")

    # converts string date time to Timestamp
    dv.date_hour = pd.to_datetime(dv.date_hour, format="%Y-%m-%d %H %M")
    dv = dv.drop(columns = ['Hour_of_day', 'Date', 'minute'])
    dv = dv.sort_values(by = ['Location_Name', 'date_hour'])

    print("date_hour to datetime done and unnecessary columns removed")

    dv = dv.drop_duplicates()

    print("duplicates dropped")

    ##### RESAMPLING CODE GOT DELETED
    # maybe use dv[dv.isnull()]==None for ease of work...???
    return None

def voltage_events(dm):
    dg = dm.copy()
    # add some missing timestamps by minute level resampling
    dg = dg.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dg['v_class'] = dg.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")
    #-------------voltage events group formations for all sites---------
    # diff_to_previous = df.v_class != df.v_class.shift(1)
    # df['v_group'] = diff_to_previous.cumsum()
    # assigns a group number to a continuous v_class event. group number
    # changes when v_class changes or when the site name changes.
    # I checked. Location transitions are taken into account while forming groups
    dg['new_group'] = dg.groupby(['Location_Name']).v_class.apply(lambda x: x != x.shift(1)).cumsum()
    #-------------------voltage event duration-------------------------
    # create a dataframe with start and end timestamp of every voltage class
    # event. We group by using the new_group column.
    df = dg.groupby(['Location_Name', 'v_class', 'new_group']).agg({'date_hour':['min', 'max'], 'voltage':['min', 'mean', 'max']})
    # rename multi level columns
    new_names={'date_hour_min':'min_date_hour', 'date_hour_max':'max_date_hour','voltage_min':'v_min','voltage_mean':'v_mean', 'voltage_max':'v_max'}
    df.columns = df.columns.map('_'.join).to_series().map(new_names)
    # calculate duration in minutes. Adding 60 seconds because we do
    # need to include both edges of date_hour.
    df = df.reset_index()
    df['dur_minute'] = ((df.max_date_hour - df.min_date_hour).dt.total_seconds() + 60.0)/60.0
    df['dur_hour'] = df['dur_minute']/60.0
    df['event_date'] = df['min_date_hour'].dt.date
    # converting dates to strings for better plot labels
    df['event_date_str'] = df.event_date.apply(lambda x: x.strftime("%d %b, %Y"))
    # code.interact(local = locals())
    return df

def count_of_abnormal_events(db):
    db = db.groupby(['Location_Name', 'v_class', 'event_date', 'event_date_str']).dur_hour.count().reset_index()
    db = db.rename(columns = {'dur_hour':'event_count'})
    return db

def voltage_summary_table(dc, db, volt_event):
    #-------------------------------------------------------
    # summarize overall data availability per site
    dv = dc.copy()
    dv = dv.groupby(['Location_Name']).resample('1D', on='date_hour').count().voltage.reset_index()
    dv = dv.rename(columns = {'voltage':'data_avail'})
    dv.data_avail = dv.data_avail.apply(lambda x: x/1440.0) #converting to days
    dv = dv.groupby(['Location_Name']).sum().data_avail.reset_index()
    dv = dv.round(2)
    #-------------------------------------------------------
    # summarize days with abnormal voltage event - "low" or "over" based on input
    d_ve = db[(db.v_class == volt_event)]
    d_ve = d_ve.groupby(['Location_Name']).sum().dur_hour.reset_index()
    d_ve = d_ve.rename(columns = {'dur_hour':'event_duration'})
    d_ve.event_duration = d_ve.event_duration.apply(lambda x: x/24.0) #converting to days
    d_ve = d_ve.round(2)
    #-------------------------------------------------------
    # long and short duration voltage outages classification
    ds = db[(db.v_class == volt_event)]
    ds['v_dur_type'] = ds.dur_hour.apply(lambda x: "long" if x>1.0 else "short")
    ds = ds.groupby(['Location_Name', 'v_dur_type']).agg({'dur_hour':['min', 'mean', 'max', 'sum', 'count']})
    new_names = {'dur_hour_min':'min_duration', 'dur_hour_mean':'mean_duration','dur_hour_max':'max_duration','dur_hour_sum':'total_duration', 'dur_hour_count':'event_count'}
    ds.columns = ds.columns.map('_'.join).to_series().map(new_names)
    ds.total_duration = ds.total_duration.apply(lambda x: x/24.0)
    ds = ds.round(2)
    ds = ds.unstack()
    #-------------------------------------------------------
    # joining all the dataframes
    dj = dv.set_index('Location_Name').join([d_ve.set_index('Location_Name'), ds])
    dj = dj.fillna(0)
    dj = dj.reset_index()
    # code.interact(local = locals())
    #-------------------------------------------------------
    # dj.to_csv('{}_voltage_summary_table.csv'.format(volt_event))
    return dj

def voltage_mode_table(dc, db, volt_event):
    #-------------------------------------------------------
    # summarize overall data availability per site
    # dv = dc.copy()
    # dv = dv.groupby(['Location_Name']).resample('1D', on='date_hour').count().voltage.reset_index()
    # dv = dv.rename(columns = {'voltage':'data_avail'})
    # dv.data_avail = dv.data_avail.apply(lambda x: x/1440.0) #converting to days
    # dv = dv.groupby(['Location_Name']).sum().data_avail.reset_index()
    # dv = dv.round(2)
    # #-------------------------------------------------------
    # # summarize days with abnormal voltage event - "low" or "over" or "no" based on input
    # d_ve = db[(db.v_class == volt_event)]
    # d_ve = d_ve.groupby(['Location_Name']).sum().dur_hour.reset_index()
    # d_ve = d_ve.rename(columns = {'dur_hour':'event_duration'})
    # d_ve.event_duration = d_ve.event_duration.apply(lambda x: x/24.0) #converting to days
    # d_ve = d_ve.round(2)
    #-------------------------------------------------------
    # long and short duration voltage outages classification
    ds = db[(db.v_class == volt_event)]
    # sum for total outage duration over the year. mean for mean lifetime of event.
    # ds = ds.groupby(['Location_Name']).agg({'dur_hour':['count', 'mean']})
    # new_names = {'dur_hour_count':'event_count', 'dur_hour_mean':'mean_duration'}
    ds = ds.groupby(['Location_Name']).agg({'dur_hour':['count', 'sum']})
    new_names = {'dur_hour_count':'event_count', 'dur_hour_sum':'total_duration'}
    ds.columns = ds.columns.map('_'.join).to_series().map(new_names)
    ds = ds.round(2)
    # ds = ds.unstack()

    # code.interact(local = locals())
    # #-------------------------------------------------------
    # # joining all the dataframes
    # dj = dv.set_index('Location_Name').join([d_ve.set_index('Location_Name'), ds])
    # dj = dj.fillna(0)
    # dj = dj.reset_index()

    #-------------------------------------------------------
    # dj.to_csv('{}_voltage_summary_table.csv'.format(volt_event))
    return ds.reset_index()

def saifi_saidi(df):
    dg = df.copy()
    # add some missing timestamps by minute level resampling
    dg = dg.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dg['v_class'] = dg.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")
    #------------------------outage parent dataset---------------------
    dg = dg.groupby(['Area', 'v_class', 'date_hour']).Location_Name.count().reset_index()
    # dg['ng1'] = dg.groupby(['Area', 'v_class']).date_hour.apply(lambda x: ((x - x.shift(-1)).abs() == pd.Timedelta('1Min')) | ((x.shift(1) - x).abs() == pd.Timedelta('1Min')))
    dg['ng2'] = dg.groupby(['Area', 'v_class']).date_hour.diff() == pd.Timedelta('1Min')
    # BETTER WAY: Assign a group everytime ng2 is False. Change the group as soon as it becomes False again.
    # so we say form a group when ng2 value is False and increment the value when it becomes False again.
    # each group can now be classified as an outage.
    dg['new_group'] = dg.groupby(['Area', 'v_class']).ng2.apply(lambda x: x==False).cumsum()
    #-----------------------outage summary dataset----------------------
    # for every outage let's select the maximum number of customers affected
    do = dg.copy()
    do = do.rename(columns = {'Location_Name':'no_cust'})
    do = do.groupby(['Area', 'v_class', 'new_group']).agg({'date_hour':['min', 'max'], 'no_cust': 'max'})
    new_names = {'date_hour_min':'min_duration', 'date_hour_max':'max_duration', 'no_cust_max':'no_cust'}
    do.columns = do.columns.map('_'.join).to_series().map(new_names)
    do = do.reset_index()
    do['duration'] = do.max_duration - do.min_duration

    # code.interact(local = locals())
    return None

def saifi_saidi_with_cust_names(df):
    def makelist(x):
        T = tuple(x)
        if len(T) > 1:
            return T
        else:
            return T[0]

    dg = df.copy()
    # add some missing timestamps by minute level resampling
    dg = dg.groupby(['Area', 'Income_Level','Location_Name']).resample('1Min', on='date_hour').mean().reset_index()
    dg['v_class'] = dg.voltage.apply(lambda x: "no" if x<101 else "low" if x>=101 and x<226 else "normal" if x>=226 and x<254 else "over" if x>=254 else "no_data")
    #------------------------outage parent dataset-----------------------
    dg = dg.groupby(['Area', 'v_class', 'date_hour']).aggregate({'Location_Name': makelist}).reset_index()
    #### Try to see if we can classify an event as outage if
    #### (1) timestamps are not conitnuous (we have that in place)
    #### OR
    #### (2) Location tuples are different
    # code.interact(local = locals())
    dg['ng2'] = dg.groupby(['Area', 'v_class']).date_hour.diff() == pd.Timedelta('1Min')
    # BETTER WAY: Assign a group everytime ng2 is False. Change the group as soon as it becomes False again.
    # so we say form a group when ng2 value is False and increment the value when it becomes False again.
    # each group can now be classified as an outage.
    dg['new_group'] = dg.groupby(['Area', 'v_class']).ng2.apply(lambda x: x==False).cumsum()
    #-----------------------outage summary dataset----------------------
    # for every outage let's select the maximum number of customers affected
    do = dg.copy()
    do = do.rename(columns = {'Location_Name':'no_cust'})
    do = do.groupby(['Area', 'v_class', 'new_group']).agg({'date_hour':['min', 'max'], 'no_cust': 'max'})

    return None

def voltage_event_clustering(db, label_data, x, y, x_label, y_label, title):
    db = db.copy()
    data = [go.Scatter(x = db[x].values, y = db[y].values, text = db[label_data], mode='markers', marker = dict(size = 10))]

    layout = go.Layout(
                       width = 1200,
                       height = 600,
                       title=title,
                       yaxis=dict(title=y_label),
                       xaxis=dict(title=x_label)
                    )

    fig = go.Figure(data=data, layout=layout)

    return py.offline.plot(
        fig,
        show_link=False,
        output_type='div',
        include_plotlyjs=False
    )

def voltage_event_clustering_poster(db, label_data, x, y, x_label, y_label, title):
    db = db.copy()
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5,3))
    markers = ["o", "^", "s", "P"]
    i = 0
    for il in db['Income_Level'].unique():
        ax.scatter(x = db[(db.Income_Level == il)][x].values, y = db[(db.Income_Level == il)][y].values, label = il, marker = markers[i])
        i = i+1
    ax.legend()
    plt.locator_params(nbins=4)
    plt.title(title)
    ax.grid(which='both', alpha=0.2)
    plt.show()
    return None

# def voltage_event_clustering_poster2(db, label_data, x, y, x_label, y_label, title):
#     db = db.copy()
#     sns.set(style="whitegrid")
#     fig, ax = plt.subplots(figsize=(6,3))
#     for il in db['Income_Level'].unique():
#         ax.scatter(x = db[(db.Income_Level == il)][x].values, y = db[(db.Income_Level == il)][y].values, label = il)
#     ax.legend()
#     plt.locator_params(nbins=4)
#     plt.title(title)
#     ax.grid(which='both', alpha=0.2)
#     plt.show()
#     return None


def stacked_bar_chart_generic(df, v_class, x, y, x_label, y_label, title):
    data = []
    for v in df[v_class].unique():
        data += [go.Bar(name=v, x=df[x].unique(), y=df[(df.v_class == v)][y])]

    layout = go.Layout(
                       width = 1200,
                       height = 600,
                       title=title,
                       yaxis=dict(title=y_label),
                       xaxis=dict(title=x_label),
                       barmode = 'stack'
                    )

    fig = go.Figure(data=data, layout=layout)

    return py.offline.plot(
        fig,
        show_link=False,
        output_type='div',
        include_plotlyjs=False
    )


def table_generic(dj, title):
    layout = go.Layout(
                   width = 1200,
                   height = 1200,
                   title=title,
                )

    table_data=[go.Table(header=dict(
                               values=list(dj.columns),
                               align='left',
                               fill_color='grey',
                               line_color='darkslategray',
                               font=dict(color='white', size=12)
                            ),
                    cells=dict(
                               values=[dj[k].tolist() for k in dj.columns[0:]],
                               align='left',
                               line_color='darkslategray',
                               font = dict(size = 11)
                            ))
                ]
    fig = go.Figure(data = table_data, layout = layout)

    return py.offline.plot(
        fig,
        show_link=False,
        output_type='div',
        include_plotlyjs=False
    )

def boxplot_button_event_specific(db, volt_event, x_label, y_label, filter_by, sortby, x_title, y_title, title, plot_type):
    dc = db.copy()
    dc = dc[(dc.v_class == volt_event)]
    ### Assemble data for multiple box plots
    if plot_type == "box":
        traces = []
        for name in sorted(dc[filter_by].unique()):
            traces.append(go.Box(
                                    x=dc[dc[filter_by] == name].sort_values(by = [sortby])[x_label],
                                    y=dc[dc[filter_by] == name].sort_values(by = [sortby])[y_label],
                                    name=name,
                                    showlegend=False,
                                    boxpoints = 'outliers'
                                ))
    else:
        traces = []
        for name in sorted(dc[filter_by].unique()):
            traces.append(go.Bar(
                                    x=dc[dc[filter_by] == name].sort_values(by = [sortby])[x_label],
                                    y=dc[dc[filter_by] == name].sort_values(by = [sortby])[y_label],
                                    name=name,
                                    showlegend=False
                                ))

    ### Create buttons for drop down menu
    labels = list(sorted(dc[filter_by].unique()))
    buttons = []
    for i, label in enumerate(labels):
        visibility = [i==j for j in range(len(labels))]
        button = dict(
                     label =  label,
                     method = 'update',
                     args = [{'visible': visibility}]
                     )
        buttons.append(button)

    updatemenus = list([
        dict(active=-1,
             x=-0.15,
             buttons=buttons
        )
    ])

    layout = go.Layout(
                       width = 1200,
                       height = 600,
                       title=title,
                       updatemenus = updatemenus,
                       yaxis=dict(title=y_title),
                       xaxis=dict(title=x_title)
                    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_xaxes(tickangle=45)

    return py.offline.plot(
        fig,
        show_link=False,
        output_type='div',
        include_plotlyjs=False
    )

def cdf_plots(df, col, x_label, y_label, title):
    sorted_data = np.sort(df[col])
    yvals = np.arange(len(sorted_data))*100.0/float(len(sorted_data) - 1)

    data = [go.Scatter(x = sorted_data, y = yvals, mode = 'lines')]

    layout = go.Layout(
                       width = 1200,
                       height = 600,
                       title=title,
                       yaxis=dict(title=y_label),
                       xaxis=dict(title=x_label)
                    )

    fig = go.Figure(data=data, layout=layout)

    return py.offline.plot(
        fig,
        show_link=False,
        output_type='div',
        include_plotlyjs=False
    )

#-----------------------HULL Plot---------------------------------------
def hull_plotting(dh, title):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6,4))
    plot_style = ['r-', 'g-.', 'b--', 'k:']
    colors = ['r', 'g', 'b', 'k']
    j = 0
    for i in dh.Income_Level.unique():
        # plt.scatter(dh[(dh.Income_Level == i)][['total_duration']], dh[(dh.Income_Level == i)][['event_count']], c=colors[j], alpha=0.8)
        # plot total duration
        points = dh[(dh.Income_Level == i)][['total_duration', 'event_count']].values
        hull = ConvexHull(dh[(dh.Income_Level == i)][['total_duration', 'event_count']])
        for simplex in hull.simplices:
            #Draw a black line between each
            plt.plot(points[simplex, 0], points[simplex, 1], plot_style[j], label = i)
        plt.scatter(162, 16.5, marker = 'x', c='k', s=100, zorder = 10)
        # remove duplicate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # fill the polygon
        plt.fill(points[hull.vertices,0], points[hull.vertices,1], colors[j], alpha=0.4)
        j = j+1
    plt.title(title)
    plt.show()
    return None


def box_plots_poster(df, dm, title):
    d_i = pd.DataFrame()
    d_i = df[['Location_Name', 'Income_Level']].drop_duplicates()
    dm = pd.merge(dm, d_i, on='Location_Name')
    dm['Income_Level'] = dm.Income_Level.apply(lambda x: "UMI" if x == "Upper Middle Income" else "LMI" if x == "Lower Middle Income" else "LI" if x == "Low Income" else "HI")
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.boxplot(x="Income_Level", y="dur_hour", hue = "problem", data = dm, showfliers = True, order=["LI", "LMI", "UMI", "HI"])
    # remove legend box title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    plt.show()
    return None

#-----------------------------------------------------------------------
def render(tpl_path, **context):
    path, filename = os.path.split(tpl_path)
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(path or '../templates'))
    template = jinja_env.get_template(filename)
    return template.render(**context)

if __name__ == '__main__':
    # create_sqlitedb()
    #------extract db and dump to pickle---------
    # df = fetch_data()
    # outfile = open('esmi_kenya_column_db', 'wb')
    # pickle.dump(df, outfile)
    # outfile.close()
    #----------Unpickle--------------------------
    infile = open(re + 'esmi_kenya_column_db', 'rb')
    df = pickle.load(infile)
    infile.close()
    # code.interact(local = locals())

    #----------Voltage events analysis dataset-----------
    df_v = voltage_events(df.copy())
    code.interact(local = locals())
    # df_saidi_wo_names = saifi_saidi(df.copy())
    # df_saidi_w_names = saifi_saidi_with_cust_names(df.copy())

    #---------counting number of occurences--------------
    # df_ev_ct = count_of_abnormal_events(df_v.copy())
    # code.interact(local = locals())

    #----------Voltage Summary Tables--------------------
    # dv_low = voltage_summary_table(df.copy(), df_v.copy(), "low")
    # dv_over = voltage_summary_table(df.copy(), df_v.copy(), "over")
    # dv_no = voltage_summary_table(df.copy(), df_v.copy(), "no")
    # code.interact(local = locals())

    # ---------Poster plots--------------------------------------------------
    dv_low = voltage_mode_table(df.copy(), df_v.copy(), "low")
    dv_over = voltage_mode_table(df.copy(), df_v.copy(), "over")
    dv_no = voltage_mode_table(df.copy(), df_v.copy(), "no")

    d_i = pd.DataFrame()
    d_i = df[['Location_Name', 'Income_Level']].drop_duplicates()
    # Add income level details to the dataset
    df_low = pd.merge(dv_low, d_i, on='Location_Name')
    df_over = pd.merge(dv_over, d_i, on='Location_Name')
    df_no = pd.merge(dv_no, d_i, on='Location_Name')
    # code.interact(local = locals())

    df_low['Income_Level'] = df_low.Income_Level.apply(lambda x: "UMI" if x == "Upper Middle Income" else "LMI" if x == "Lower Middle Income" else "LI" if x == "Low Income" else "HI")
    df_over['Income_Level'] = df_over.Income_Level.apply(lambda x: "UMI" if x == "Upper Middle Income" else "LMI" if x == "Lower Middle Income" else "LI" if x == "Low Income" else "HI")
    df_no['Income_Level'] = df_no.Income_Level.apply(lambda x: "UMI" if x == "Upper Middle Income" else "LMI" if x == "Lower Middle Income" else "LI" if x == "Low Income" else "HI")

    # hull_plotting(df_low)
    # hull_plotting(df_over)
    # hull_plotting(df_no)

    # Get distribution of mean event life time and event frequency
    dg = df_v.copy()
    dg = dg[(dg.v_class != "normal") & (dg.v_class != "no_data")]
    dg['problem'] = dg.v_class.apply(lambda x: "No Supply" if x=="no" else "Poor PQ")
    dg_dur = dg.groupby(['Location_Name', 'problem']).mean()[['dur_hour']].reset_index()
    dg_count = dg.groupby(['Location_Name', 'problem']).count()[['dur_hour']].reset_index()
    # box_plots_poster(df.copy(), dg_dur.copy(), "Mean Event Lifetime (hours)")
    # box_plots_poster(df.copy(), dg_count.copy(), "Total Event Count")
    code.interact(local = locals())

    # voltage_event_clustering_poster(df_low.copy(), 'Location_Name', 'mean_duration', 'event_count', "Mean LV Event Duration", "Total LV Event Count", "Low Voltage Events")
    # voltage_event_clustering_poster(df_over.copy(), 'Location_Name', 'mean_duration', 'event_count', "Mean LV Event Duration", "Total LV Event Count", "Over Voltage Events")
    # voltage_event_clustering_poster(df_no.copy(), 'Location_Name', 'mean_duration', 'event_count', "Mean LV Event Duration", "Total LV Event Count", "No Supply Events")

    # voltage_event_clustering_poster(df_low.copy(), 'Location_Name', ('mean_duration', 'long'), ('event_count', 'long'), "Mean Long LV Event Duration", "Total Long LV Event Count", "Long Duration LV Events")
    # voltage_event_clustering_poster(df_low.copy(), 'Location_Name', ('mean_duration', 'short'), ('event_count', 'short'), "Mean Short LV Event Duration", "Total Short LV Event Count", "Short Duration LV Events")

    # voltage_event_clustering_poster(df_over.copy(), 'Location_Name', ('mean_duration', 'long'), ('event_count', 'long'), "Mean Long OV Event Duration", "Total Long OV Event Count", "Long Duration OV Events")
    # voltage_event_clustering_poster(df_over.copy(), 'Location_Name', ('mean_duration', 'short'), ('event_count', 'short'), "Mean Short OV Event Duration", "Total Short OV Event Count", "Short Duration OV Events")

    # voltage_event_clustering_poster(df_no.copy(), 'Location_Name', ('mean_duration', 'long'), ('event_count', 'long'), "Mean Long LV Event Duration", "Total Long LV Event Count", "Long Duration No Supply Events")
    # voltage_event_clustering_poster(df_no.copy(), 'Location_Name', ('mean_duration', 'short'), ('event_count', 'short'), "Mean Short LV Event Duration", "Total Short LV Event Count", "Short Duration No Supply Events")

    #---------HTML report plots---------------------------
    plots = []
    # voltage_event_clustering(db, label_data, x, y, x_label, y_label, title)
    plots += [voltage_event_clustering(dv_low.copy(), 'Location_Name', ('mean_duration', 'long'), ('event_count', 'long'), "Mean Long LV Event Duration", "Total Long LV Event Count", "Long-lived Low Voltage Event Count Versus Duration")]
    plots += [voltage_event_clustering(dv_low.copy(), 'Location_Name', ('mean_duration', 'short'), ('event_count', 'short'), "Mean Short LV Event Duration", "Total Short LV Event Count", "Short-lived Low Voltage Event Count Versus Duration")]
    plots += [voltage_event_clustering(dv_over.copy(), 'Location_Name', ('mean_duration', 'long'), ('event_count', 'long'), "Mean Long OV Event Duration", "Total Long OV Event Count", "Long-lived Over Voltage Event Count Versus Duration")]
    plots += [voltage_event_clustering(dv_over.copy(), 'Location_Name', ('mean_duration', 'short'), ('event_count', 'short'), "Mean Short OV Event Duration", "Total Short OV Event Count", "Short-lived Over Voltage Event Count Versus Duration")]

    # plots +=[stacked_bar_chart_generic(df_saidi.reset_index().copy(), 'v_class', 'Area', 'saidi', 'Areas', 'saidi', "No Supply, Low Voltage, and Over Voltage SAIDI")]
    #---------Plotting the CDFs of voltage Durations----
    plots += [cdf_plots(dv_low.copy(), 'event_duration', "Total Low Voltage Duration (Days)", "(%) of Customers", "CDF of Customers by Total Low Voltage Duration")]
    plots += [cdf_plots(dv_over.copy(), 'event_duration', "Total Over Voltage Duration (Days)", "(%) of Customers", "CDF of Customers by Total Over Voltage Duration")]
    plots += [cdf_plots(dv_no.copy(), 'event_duration', "Total No Supply Duration (Days)", "(%) of Customers", "CDF of Customers by Total No Supply Duration")]

    #---------Plotting Voltage Duration Plots-----------
    plots += [boxplot_button_event_specific(df_v.copy(), "low", "event_date_str", "dur_hour", "Location_Name", "min_date_hour", "Outage Date", "Outage Duration (hours)", "Low Voltage Event Duration", "box")]
    plots += [boxplot_button_event_specific(df_v.copy(), "over", "event_date_str", "dur_hour", "Location_Name", "min_date_hour", "Outage Date", "Outage Duration (hours)", "Over Voltage Event Duration", "box")]

    #----------Plotting Abnormal Event Count------------
    plots += [boxplot_button_event_specific(df_ev_ct.copy(), "low", "event_date_str", "event_count", "Location_Name", "event_date", "Outage Date", "Outage Count", "Low Voltage Event Count", "bar")]
    plots += [boxplot_button_event_specific(df_ev_ct.copy(), "over", "event_date_str", "event_count", "Location_Name", "event_date", "Outage Date", "Outage Count", "Over Voltage Event Count", "bar")]

    #----------Render HTML Report-----------------------
    html = render(
        CURRENTDIR + '/esmi_kenya_voltage_events.html',
        report_title='ESMI Kenya Abnormal Voltage Events Report',
        full_period_str='',
        screen_width=1200,
        graph_list=plots,
        now_str='',
        support_email='zshah@umass.edu'
    )
    open(
        CURRENTDIR + '/esmi_kenya_voltage_report3.html',
        'w'
    ).write(html)