import netCDF4
import numpy as np
import cPickle as pickle
import multiprocessing as mp

def get_precip_series(ilong, ilat, j):
    precip = []
    for yr in yrs: 
        for d in range(1,ds[j]+1):
            date_stamp = str(yr) + ms[j] + str(d).zfill(2)
            nc_file = '../TRMM_data/3B42_Daily.'+date_stamp+'.7.nc4.nc4'
            f = netCDF4.Dataset(nc_file, mode='r')
            precip_val = f.variables['precipitation'][ilong,ilat]
            precip.append(precip_val)
    return np.array(precip)

def get_lambda1(precip):
    wetdays = len(precip[precip>0.0])
    return float(wetdays)/len(precip)

def get_lambda2(precip):
    inonzero = np.where(precip>0.0)[0]
    intervals = np.diff(inonzero)
    return 1.0/np.mean(intervals) # do we need to subtract 1.0 here to account for consecutive rain days, NO??

def get_alpha(precip):
    return np.mean(precip[precip>0.0])

def evaluate_precip_params(lon_arr):
    params_arr = np.zeros((maxlon, maxlat, 12, 2))
    for i, lo in enumerate(lon_arr):
        for j, la in enumerate(lat_arr):
            for k in np.arange(12):
                print lo,la,k
                precip = get_precip_series(lo,la,k)
                params_arr[i,j,k,0] = get_lambda1(precip)
                params_arr[i,j,k,1] = get_alpha(precip)
    return params_arr

def main():
    ncores = mp.cpu_count(); print('There are %s cores on this machine '%(str(ncores),))
    pool = mp.Pool()
    lon_cores = np.array_split(lon_arr, ncores)
    params_pooled = pool.map(evaluate_precip_params, lon_cores) # will output four arrays based on input lon_cores
    params = np.vstack(params_pooled)
    with open('./lambda_alphas.pickle', 'wb') as output_file: 
        pickle.dump(params, output_file)
    return params

if __name__ == "__main__":
    ms = ['01','02','03','04','05','06','07','08','09','10','11','12']
    ds = [31,28,31,30,31,30,31,31,30,31,30,31] # how to incorporate leap years? 
    yrs = np.arange(1998, 2016)
    maxlat, maxlon = 400, 1440
    lon_arr = np.arange(0, maxlon) # lon_arr are split equally amongst the cores 
    lat_arr = np.arange(0, maxlat)
    params = main()
    