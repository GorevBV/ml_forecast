
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import lightgbm as lgb

from pymongo import MongoClient
from netCDF4 import Dataset
from shutil import copyfile
from sklearn.metrics import mean_squared_error as mse

import datetime
import time
import os
import copy
import math
import gc


# In[4]:

def spherical_metric(points1, points2, Rad = 6371):
    """
    Returns array of distances between two lists of geopoints
    """
    
    points1 = np.radians(points1)
    points2 = np.radians(points2)
    metrics = np.ndarray(shape = (len(points1), len(points2)))

    for p in range(len(points1)):

        for q in range(len(points2)):
            
            metrics[p][q] = Rad * 2 * math.asin(math.sqrt(math.sin((points2[q][1]-points1[p][1]) / 2) ** 2 + 
                                                   math.cos(points1[p][1]) * math.cos(points2[q][1]) * 
                                                   math.sin((points2[q][0]-points1[p][0]) / 2) ** 2))


    return metrics


# In[6]:

def load_stations(mng_stations, xlong, xlat, E_Rad, r_last, border = [5, 5]):
    """
    Returns: two dataframes: stations(1) located in grid and stations(2) located within r_last radius of (1);
             one array: array of distances between (1) and (2)
    """
    x0, y0 = border
    xlong = xlong[x0:-x0, y0:-y0]
    xlat = xlat[x0:-x0, y0:-y0]
    
    nlon, nlat = xlat.shape
    grid = np.dstack([xlong, xlat]).reshape(nlat * nlon, 2)
    
    indexes = []
    
    for i in range(nlon):
        for j in range(nlat):
            indexes.append((i + x0, j + y0))
    
    
    stations2 = pd.DataFrame(columns = ['id', 'coordinates'])

### mng_stat is a cursor for mongo directory with station files
### mng_stat = MongoClient("mongodb://136.243.82.68:27000/").sources.mem.find({})

    for doc in mng_stations:
        stations2.loc[len(stations2)] = [doc['_id'], doc['location']['coordinates']]
    
### collect stations' coordinates to define their long-lat grid location

    coordinates = stations2['coordinates'].values.tolist()
    
#     indexes, stations_grid = my_interpol(grid, xlong, xlat)
#     stations2['indexes'] = indexes
    
### drop unused columns

    stations2.drop(['coordinates'], axis = 1, inplace = True)
    
### remove out-of-grid stations    

#     stations1 = stations2.loc[~stations2['indexes'].isin([(-1, -1)])]
    
### definition of out-of-range stations

    dist_list = spherical_metric(grid, coordinates, E_Rad)
    
#     stations1.reset_index(inplace = True)
    
#     for index, row in stations1.iterrows():
#         dist_list[index][row[0]] = np.infty
    
    stations1 = pd.DataFrame(columns = ['indexes', 'names'])
    stations1['indexes'] = indexes
    stations1['names'] = indexes
    stations1.reset_index(inplace = True)
    stations1.rename(columns = {"index": "id"}, inplace = True)
    # stations1 = stations1[['id', 'indexes', 'names']]
    
    stations2['indexes'] = np.min(dist_list, axis = 0) <= r_last
    stations2.reset_index(inplace = True)
    # stations2['Num'] = range(stations2.shape[0])
    stations2 = stations2.loc[~stations2['indexes'].isin([False])]
    stations2 = stations2.drop(['indexes'], axis = 1)
    stations2.rename(columns = {"id": "names", "index": "id"}, inplace = True)

    
    return stations1, stations2, dist_list


# In[8]:

def selecting_square(points, layers = 1):
    """
    :param points: (xlon, xlat)
    :param layers: square size
    :return: edges of (1 + 2 * layers) x (1 + 2 * layers) square
    """
    row_start = points[0] - layers
    row_end = points[0] + layers + 1

    col_start = points[1] - layers
    col_end = points[1] + layers + 1
    
    
    return row_start, row_end, col_start, col_end


# In[9]:

def get_df(some_arr, feat_name, present = False, backtrack = 2,  size = 9): # pres, tb, size - mongo
    """
    :param some_arr: array with columns inside column
    :param feat_name: name of that column
    :param present: indicator of subcolumn with zero index
    :param backtrack: number of columns inside feat_name column
    :param size: number of columns inside feat_name column
    Returns dataframe with necessary features
    """
    cols = ['id', 'Datetime', 'timedelta']
    
    for h in range(backtrack):
        cols.append(feat_name + '_prev' + str(backtrack - h))
    
    if present:
        cols.append(feat_name)
    
    train = pd.DataFrame(columns = cols, data = some_arr)
    
    for column in cols[3:]:
        splitted_co = train[column].apply(pd.Series)
        new_names = []
        for i in range(int((1 - size) / 2), int(size / 2) + 1):
            new_names.append(column + '_' + str(i))
        
        splitted_co.columns = new_names
        train = pd.concat([train, splitted_co], axis = 1)
        train.drop(column, axis = 1, inplace = True)
    
        
    return train


# In[10]:

def bias(df1, df2):
    """
    Returns Bias error between df1 and df2 vectors
    """
    
    accuracy = 0
    
    for j,z in zip(df1.ix[:, 0], df2.ix[:, 0]):
        
        accuracy += (j - z)
            
    return accuracy/df2.shape[0]

def rmse(y_pred, y_true):
    
    return np.sqrt(mse(y_pred, y_true))


# In[11]:

def time_from_name(name):
    """
    :param name: string like 'word1_word2_year-month-day_hour:minute:second'
    Returns time and date in datetime format
    """
    
    yr = int(name.split('_')[2].split('-')[0])
    mth = int(name.split('_')[2].split('-')[1])
    day = int(name.split('_')[2].split('-')[2])
    hr = int(name.split('_')[3].split(':')[0])
    
    return datetime.datetime(yr, mth, day, hr)


# In[12]:

def distance_to_list(distt, r_list, r_min, L_):
    """
    :distt: array of distances between 2 sets of geopoints
    Returns 2 lists of distances and indexes of axis-1 geopoints that are within r-* radius of axis-0 geop-ts
    """

    r_num = len(r_list)
    startt = time.time()
    # print('old r_list: ', r_list)
    r_list = [r_min] + r_list
    # print('new r_list: ', r_list)

    w_list = []
    for r in range(r_num):
        w_list.append([])
    
    for i in range(distt.shape[0]):
        
        for r in range(r_num):
            w_list[r].append([])
        
        
        for j in range(distt[i].shape[0]):

            if distt[i][j] <= r_min:
                w = math.exp(- 0.5 * (r_min / L_) ** 2) # w = 1.0 / r_min
                for r in range(r_num):
                    w_list[r][i].append([w, j])

            for rn in range(r_num):
                if (distt[i][j] > r_list[rn]) and (distt[i][j] <= r_list[rn + 1]):
                    w = math.exp(- 0.5 * (distt[i][j] / L_) ** 2) # w = 1.0 / distt[i][j]
                    for r in range(rn, r_num):
                        w_list[r][i].append([w, j])

        
    print('Weight function is EXP(-0.5 * (R / L) ^ 2)')
    # print('Weight function is 1.0 / R')
    endd = time.time()
    print('!!!!Creating distance list time: ', endd - startt)
    
    return w_list


# In[14]:

# add time condition

def load_data_2017(settgs, Argument, T, time_0, filename, stat_grid): 
    """
    Returns pandas dataframe with columns like (id, Datetime, timedelta, *features*, target)
    """
    
#### Collecting model features

    targets = ['co', 'no', 'no2', 'o3', 'PM2_5_DRY', 'PM10', 'so2']
    # feats_0 = [settgs['target']['model_name']] + settgs['features0'] # ['co', 'COSZEN']
    # feats_1 = settgs['features1'] # ['U10', 'V10']
    # feats = feats_0 + feats_1

    layer = settgs['square_size']
    time_steps = settgs['backtrack'] + 1

    time_1 = T - time_0
    hour = time_1.days * 24 + int(time_1.seconds / 3600) - 1

    array_co = []
    array_no = []
    array_no2 = []
    array_o3 = []
    array_pm25 = []
    array_pm10 = []
    array_so2 = []
    array_coszen = []
    
    print('Starting model features')

    data = Dataset(filename, 'r')

    for index, row in stat_grid.iterrows():

    	row_start, row_end, col_start, col_end = selecting_square([row[1][0], row[1][1]], 
                                                                  layers = layer)

        id_station = row[0]

        values_co = []
        values_no = []
        values_no2 = []
        values_o3 = []
        values_pm25 = []
        values_pm10 = []
        values_so2 = []
        values_coszen = []

        for i in range(time_steps):
            values_co.append([])
            values_no.append([])
            values_no2.append([])
            values_o3.append([])
            values_pm25.append([])
            values_pm10.append([])
            values_so2.append([])
            values_coszen.append([])

        for inside_row in range(row_start, row_end):

            ts_1 = hour - time_steps + 1
            for m in range(time_steps):
            	values_co[m] += ([x for x in data['co'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_no[m] += ([x for x in data['no'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_no2[m] += ([x for x in data['no2'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_o3[m] += ([x for x in data['o3'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_pm25[m] += ([x for x in data['PM2_5_DRY'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_pm10[m] += ([x for x in data['PM10'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_so2[m] += ([x for x in data['so2'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_coszen[m] += ([x for x in data['COSZEN'][ts_1 + m, inside_row, col_start:col_end]])


        to_append = [id_station, T, hour + 1]
        array_co.append(to_append + values_co)
        array_no.append(to_append + values_no)
        array_no2.append(to_append + values_no2)
        array_o3.append(to_append + values_o3)
        array_pm25.append(to_append + values_pm25)
        array_pm10.append(to_append + values_pm10)
        array_so2.append(to_append + values_so2)
        array_coszen.append(to_append + values_coszen)

        del values_co[:], values_no[:], values_no2[:], values_o3[:], values_pm25[:], values_pm10[:], values_so2[:], values_coszen[:]


    data.close()
    
    sze = (1 + 2 * layer) ** 2

    temp_coszen = get_df(array_coszen, 'COSZEN', present = True, backtrack = time_steps - 1, size = sze)

    temp_co = get_df(array_co, 'co', present = True, backtrack = time_steps - 1, size = sze)
    temp_no = get_df(array_no, 'no', present = True, backtrack = time_steps - 1, size = sze)
    temp_no2 = get_df(array_no2, 'no2', present = True, backtrack = time_steps - 1, size = sze)
    temp_o3 = get_df(array_o3, 'o3', present = True, backtrack = time_steps - 1, size = sze)
    temp_pm25 = get_df(array_pm25, 'PM2_5_DRY', present = True, backtrack = time_steps - 1, size = sze)
    temp_pm10 = get_df(array_pm10, 'PM10', present = True, backtrack = time_steps - 1, size = sze)
    temp_so2 = get_df(array_so2, 'so2', present = True, backtrack = time_steps - 1, size = sze)

    del array_co[:], array_no[:], array_no2[:], array_o3[:], array_pm25[:], array_pm10[:], array_so2[:], array_coszen[:]

    temp_co = pd.merge(temp_coszen, temp_co, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_no = pd.merge(temp_coszen, temp_no, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_no2 = pd.merge(temp_coszen, temp_no2, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_o3 = pd.merge(temp_coszen, temp_o3, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_pm25 = pd.merge(temp_coszen, temp_pm25, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_pm10 = pd.merge(temp_coszen, temp_pm10, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_so2 = pd.merge(temp_coszen, temp_so2, how = 'inner', on = ['id','Datetime','timedelta'])
        
#### Collecting emission data during dt1-dt2 period

    print('Starting emissions')

    filepath_wc = "/home/wrf/data/emissions/" + Argument + "/"

    files = [x for x in os.listdir(filepath_wc) if (('wrfchemi_d02' in x) and (time_from_name(x) == time_0))]
    files.sort()
    file_wc = files[-1]

    time_steps = settgs['backtrack']
    hour += 1

    array_co = []
    array_no = []
    array_no2 = []
    # array_o3 = []
    array_pm25 = []
    array_pm10 = []
    array_so2 = []
    
    data = Dataset(os.path.join(filepath_wc, file_wc), 'r') # emission = 
    # data = emission.variables[settgs['target']['emiss_name']][:]
    
    for index, row in stat_grid.iterrows():

        id_station = row[0]
        row_start, row_end, col_start, col_end = selecting_square([row[1][0], row[1][1]], 
                                                                 layers = layer)

        values_co = []
        values_no = []
        values_no2 = []
        values_pm25 = []
        values_pm10 = []
        values_so2 = []

        for i in range(time_steps):
            values_co.append([])
            values_no.append([])
            values_no2.append([])
            values_pm25.append([])
            values_pm10.append([])
            values_so2.append([])

        for inside_row in range(row_start, row_end):
            ts_1 = hour - time_steps
            for m in range(time_steps):
            	values_co[m] += ([x for x in data['E_CO'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_no[m] += ([x for x in data['E_NO'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_no2[m] += ([x for x in data['E_NO2'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	# values_o3[m] += ([x for x in data['o3'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_pm25[m] += ([x for x in data['E_PM_25'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_pm10[m] += ([x for x in data['E_PM_10'][ts_1 + m, 0, inside_row, col_start:col_end]])
            	values_so2[m] += ([x for x in data['E_SO2'][ts_1 + m, 0, inside_row, col_start:col_end]])


        array_co.append([id_station, T, hour, values_co[0], values_co[1]])
        array_no.append([id_station, T, hour, values_no[0], values_no[1]])
        array_no2.append([id_station, T, hour, values_no2[0], values_no2[1]])
        # array_o3.append([id_station, T, hour, values_no2[0], values_no2[1]])
        array_pm25.append([id_station, T, hour, values_pm25[0], values_pm25[1]])
        array_pm10.append([id_station, T, hour, values_pm10[0], values_pm10[1]])
        array_so2.append([id_station, T, hour, values_so2[0], values_so2[1]])

        del values_co[:], values_no[:], values_no2[:], values_pm25[:], values_pm10[:], values_so2[:]
                
    data.close()
    
    sze = (1 + 2 * layer) ** 2

    temp0 = get_df(array_co, 'emiss', backtrack = time_steps, size = sze)
    temp_co = pd.merge(temp0, temp_co, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_co[:]

    temp0 = get_df(array_no, 'emiss', backtrack = time_steps, size = sze)
    temp_no = pd.merge(temp0, temp_no, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_no[:]

    temp0 = get_df(array_no2, 'emiss', backtrack = time_steps, size = sze)
    temp_no2 = pd.merge(temp0, temp_no2, how = 'inner', on = ['id','Datetime','timedelta'])
    temp_o3 = pd.merge(temp0, temp_o3, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_no2[:]

    temp0 = get_df(array_pm25, 'emiss', backtrack = time_steps, size = sze)
    temp_pm25 = pd.merge(temp0, temp_pm25, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_pm25[:]

    temp0 = get_df(array_pm10, 'emiss', backtrack = time_steps, size = sze)
    temp_pm10 = pd.merge(temp0, temp_pm10, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_pm10[:]

    temp0 = get_df(array_so2, 'emiss', backtrack = time_steps, size = sze)
    temp_so2 = pd.merge(temp0, temp_so2, how = 'inner', on = ['id','Datetime','timedelta'])
    del array_so2[:]


    return temp_co, temp_no, temp_no2, temp_o3, temp_pm25, temp_pm10, temp_so2


SETTINGS_MONGO = 'mongodb://root:LgQoE74G6wXmNN16W70nk6epAK65wj@136.243.90.109:27027/'
settings_mng = MongoClient(SETTINGS_MONGO)
mng = settings_mng.settings.script_settings.find_one({'script': 'ml_predict_rec.py'})

args = mng['args']
print('Args: ', args)
cities = settings_mng.settings.cities.find_one({'_id' : args[0].lower()})
if cities == None:
    source = 'mem'
else:
    source = cities['fact_sources'][0]

print('source: ', source)
print('square_size: ', mng['square_size'])
print('stat_rad: ', mng['stat_rad'])
print('weight_coef: ', mng['weight_coef'])
print('model_coef: ', mng['model_coef'])
print('boost: ', mng['boost'])
# print('target: ', mng['target'])
print('border_const: ', mng['border_const'])

ct = datetime.datetime.now()
print('Current time: ', ct)
Timepoint = datetime.datetime(ct.year, ct.month, ct.day, ct.hour)
print(Timepoint)

min_hour = mng['min_hour']


file_for_pred = "/home/wrf/data/dump/ML_" + args[0] + ".nc"

data = Dataset(file_for_pred, 'r')
xlong = data['XLONG'][0,:,:]
xlat = data['XLAT'][0,:,:]
bytes_time = data['Times'][47,:]
strtime = 'date_time_'
for ch in str_time:
    strtime += ch.decode('UTF-8')
data.close()

Time_0 = time_from_name(strtime)
Time_0 = Time_0 - datetime.timedelta(hours = 1)
# copyfile(os.path.join(filepath, files[-1]), file_for_pred)

    
brdr = [mng['border_const'][0] + mng['square_size'], mng['border_const'][1] + mng['square_size']]
data_mng = MongoClient(mng['data_mongo_path'])
mng_stat = data_mng.sources[source].find({})
inner_grid, stations_cover, dist = load_stations(mng_stat, xlong, xlat, mng['Earth_Radius'], mng['stat_rad'][-1], brdr)
#####

###########
print('Collecting observations')

tgts = ['co', 'no', 'no2', 'o3', 'pm25', 'pm10', 'so2']
tgts_mod = ['co', 'no', 'no2', 'o3', 'PM2_5_DRY', 'PM10', 'so2']
tgts_obs = ['CO', 'NO', 'NO2', 'O3', 'PM2_5_DRY', 'PM10', 'SO2'] # mng['target']['targ_name']
print('Targets: ', tgt)

max_val = 10000.0 # mng['target']['max_value']
print('MAX value: ', max_val)
    
bt1 = Timepoint - datetime.timedelta(hours = len(mng['stat_rad'])) # 2 + 3 # bordertime1 - start
bt2 = Timepoint # + datetime.timedelta(hours = 1) # bordertime2 - end
    
data_mng = MongoClient(mng['data_mongo_path'])

d_min = mng['stat_rad_min']
L = mng['weight_coef']
w_min = math.exp(- 0.5 * (mng['model_coef'] / L) ** 2)

stlist = distance_to_list(dist, mng['stat_rad'], d_min, L)
glb = globals()

for p in range(len(tgts)):

    mng_dat = data_mng.data[source].find({'$and': [{'_id.Datetime': {'$gte': bt1}}, {'_id.Datetime': {'$lt': bt2}},
                                          {tgts_obs[p]: {'$ne': None}}, {tgts_obs[p]: {'$lte': max_val}}, {tgts_obs[p]: {'$gt': 0}}]})

    
    dataset = pd.DataFrame(columns = ['names','Datetime','target_' + tgts_obs[p]])
    
    for doc in mng_dat:
    	dataset.loc[len(dataset)] = [doc['_id']['station'], doc['_id']['Datetime'], doc[tgts_obs[p]]]
    
    dataset = dataset.sort_values(by = ['Datetime','names'], ascending = [1, 1])
    # test = dataset[dataset['Datetime'] == Timepoint]
    # if test.shape[0] > 0:
    # 	print('MAX obs, MIN obs: ', max(test['target_' + tgt]), min(test['target_' + tgt]))
    # else:
    # 	print('(Datetime == Timepoint)-data is empty!')


    stations_grid = inner_grid.drop(['indexes', 'names'], axis = 1)
    dataset = pd.merge(dataset, stations_cover, how = 'inner', on = ['names'])
    dataset.drop(['names'], axis = 1, inplace = True)

    dataset = dataset[['id', 'Datetime', 'target_' + tgts_obs[p]]]
    dataset.sort_values(by = ['id', 'Datetime'], inplace = True)

	
    # dist = np.delete(dist, np.s_[:])

    for r in range(len(mng['stat_rad'])):
    	# print('LETS_GO_' + str(r + 1))
    	D11, D12 = [], []
    	dataset['Datetime'] += datetime.timedelta(hours = 1)
    	dtst1 = dataset[dataset['Datetime'] == Timepoint] # .loc
    	for ind, row in stations_grid.iterrows():
            summ, f_w = 0, 0
            for rpk in stlist[r][int(row[0])]: 
            	dtst = dtst1[dtst1['id'] == rpk[1]] # .loc
            	if dtst.shape[0] != 0:
                    summ += dtst.iloc[0, 2] * rpk[0]
                    f_w += rpk[0]
            D11.append(summ)
            D12.append(f_w)
    	stations_grid['D' + str(r + 1) + '1'] = D11
    	stations_grid['D' + str(r + 1) + '2'] = D12
    	del D11[:], D12[:]
    

    shp = stations_grid.shape[0]
    stations_grid['Datetime'] = [Timepoint] * shp
    sup_df = copy.deepcopy(stations_grid)
    for hlt in range(len(mng['stat_rad']) - 1):
        sup_df.drop(['D' + str(len(mng['stat_rad'])) + '1', 'D' + str(len(mng['stat_rad'])) + '2'], axis = 1, inplace = True)
        for fg in range(len(mng['stat_rad']) - 1, 0, -1):
            sup_df.rename(columns = {'D' + str(fg) + '1': 'D' + str(fg + 1) + '1', 'D' + str(fg) + '2': 'D' + str(fg + 1) + '2'}, inplace = True)
        sup_df['D11'] = [0] * shp
        sup_df['D12'] = [0] * shp
        sup_df['Datetime'] = [Timepoint + datetime.timedelta(hours = hlt + 1)] * shp

        col_names = []
        for r in range(1, len(mng['stat_rad']) + 1):
            col_names.append('D' + str(r) + '1')
            col_names.append('D' + str(r) + '2')
        sup_df = sup_df[['id', 'Datetime'] + col_names]
        stations_grid = pd.concat([stations_grid, sup_df], ignore_index = True)

    print('Observations are collected and measured')
    glb['stations_grid_' + tgts[p]] = copy.deepcopy(stations_grid)

del stlist[:]

####

bd = (mng['border_const'][0] + mng['square_size'], mng['border_const'][1] + mng['square_size'])
ts = Timepoint - Time_0
start_hour = ts.days * 24 + int(ts.seconds / 3600)


for hour in range(start_hour, max_hour + 1):
    t_moment = Time_0 + datetime.timedelta(hours = hour)
    x_pred_co, x_pred_no, x_pred_no2, x_pred_o3, x_pred_pm25, x_pred_pm10, \
    x_pred_so2 = load_data_2017(mng, args[0], t_moment, Time_0, file_for_pred, inner_grid)
    dump_file = Dataset(file_for_pred, 'r+')

    for p in range(len(tgts)):
        bst = lgb.Booster(model_file = mng['boost'][tgts_obs[p]])
        glb['x_pred_' + tgts[p]] = glb['x_pred_' + tgts[p]].merge(glb['stations_grid_' + tgts[p]], how = 'left', on = ['id', 'Datetime'])
        glb['x_pred_' + tgts[p]] = glb['x_pred_' + tgts[p]].sort_values(by = ['id'])
        for col in col_names:
            glb['x_pred_' + tgts[p]][col].replace(to_replace = np.nan, value = 0, inplace = True)

        for sn in range (1, len(mng['stat_rad']) + 1):
            glb['x_pred_' + tgts[p]]['D_-' + str(sn)] = (glb['x_pred_' + tgts[p]]['D' + str(sn) + '1'] + w_min * \
                glb['x_pred_' + tgts[p]][tgts_mod[p] + '_0']) / (glb['x_pred_' + tgts[p]]['D' + str(sn) + '2'] + w_min)
            glb['x_pred_' + tgts[p]].drop(['D' + str(sn) + '1', 'D' + str(sn) + '2'], axis = 1, inplace = True)

        # print('dataset ready')

        
        glb['x_pred_' + tgts[p]].drop(['Datetime', 'id'], axis = 1, inplace = True)
        y_pred = bst.predict(glb['x_pred_' + tgts[p]])
        size = y_pred.shape[0]
        size = int(np.sqrt(size))
        y_pred = y_pred.reshape(size, size)
        print(hour)

        tries = dump_file[tgts_mod[p]][:]
        tries[hour - 1][0][bd[0]:(size + bd[0]), bd[1]:(size + bd[1])] = y_pred
        dump_file.variables[tgts_mod[p]][:] = tries

    dump_file.close()