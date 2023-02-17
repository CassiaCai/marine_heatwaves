import s3fs
import xarray as xr
import numpy as np
import pandas as pd; 
import dask.array as da
import ocetrac
import warnings
import expectexception
import netCDF4 as nc
import datetime as dt
import scipy
import intake
import pprint
from skimage.measure import label as label_np, regionprops

def apply_ocetrac_to_CESM2LE_SST_field(
    member_id_val = 0, 
    threshold_val = 0.9, 
    radius_val = 3, 
    min_size_quartile_val = 0.75, 
    start_val = 0, 
    end_val = 1980):
    """
    Apply ocetrac (https://ocetrac.readthedocs.io/en/latest/) to the
    CESM2 Large Ensemble SST field, which has 100 ensemble members. 
    The two main Ocetrac parameters are radius_val and 
    min_size_quartile_val which can be tuned according to data 
    resolution and distribution.
    Parameters
    ------------
    member_id_val : int
        In the case of CESM2 LE, the member_id_val can be any value 
        from 0 to 99 inclusive. This value depends on the number of
        ensemble members.
    threshold_val : float
        Anomalies that exceed the monthly threshold_val (e.g., 90th  
        or 0.9) percentile will be considered here as extreme.
    radius_val : int
        Number of grid cells defining the width of the structuring
        element used in the morphological operations.
    min_size_quartile_val : float
        Threshold to subsample the objects according the the 
        distribution of object area.
    start_val : int
        In the case of CESM2 LE, this can be 0 to 1980 because
        1980 months (or 165 years) were simulated.
    end_val : int
        See description for start_val.
    Returns
    ------------
    detrended : xarray.DataArray (time:1980, lat: 192, lon: 288)
    blobs : xarray.DataArray 'labels' (time: 1980, lat: 192, lon: 288)
    """
    subset = coll_orig.search(component='atm',
                              variable='SST',
                              frequency='month_1',
                              experiment='historical',
                              member_id= str(member_id_list[1]))
    dsets = subset.to_dataset_dict(zarr_kwargs={"consolidated": True}, 
                                   storage_options={"anon": True})
    if member_id_val < 50:
        ds = dsets['atm.historical.cam.h0.cmip6.SST']
    else:
        ds = dsets['atm.historical.cam.h0.smbb.SST']
    SST = ds.SST.isel(member_id=0)
    SST.load()
    
    # detrending
    dyr = SST.time.dt.year + (SST.time.dt.month-0.5)/12
    dyr = dyr[start_val:end_val]
    
    # our 6 coefficient model is composed of the mean, trend, annual sine and cosine harmonics, & semi-annual sine and cosine harmonics
    model = np.array([np.ones(len(dyr))] + [dyr-np.mean(dyr)] + [np.sin(2*np.pi*dyr)] + [np.cos(2*np.pi*dyr)] + [np.sin(4*np.pi*dyr)] + [np.cos(4*np.pi*dyr)])
    
    # take the pseudo-inverse of model to 'solve' least-squares problem
    pmodel = np.linalg.pinv(model)
    model_da = xr.DataArray(model.T, 
                            dims=['time','coeff'], 
                            coords={'time':SST.time.values[start_val:end_val], 
                                    'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, 
                             dims=['coeff','time'], 
                             coords={'coeff':np.arange(1,7,1), 
                                     'time':SST.time.values[start_val:end_val]})
    
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(SST), 
                           dims=['coeff','lat','lon'], 
                           coords={'coeff':np.arange(1,7,1), 
                                   'lat':SST.lat.values, 
                                   'lon':SST.lon.values})
    
    # construct mean, trend, and seasonal cycle
    mean = model_da[:,0].dot(sst_mod[0,:,:])
    trend = model_da[:,1].dot(sst_mod[1,:,:])
    seas = model_da[:,2:].dot(sst_mod[2:,:,:])
    
    # compute anomalies by removing all  the model coefficients 
    ssta_notrend = SST-model_da.dot(sst_mod) #this is anomalies
    detrended = ssta_notrend
    if detrended.chunks:
        detrended = detrended.chunk({'time': -1})
    threshold = detrended.groupby('time.month').quantile(threshold_val,dim=('time')) 
    features_ssta = detrended.where(detrended.groupby('time.month')>=threshold, other=np.nan)
    features_ssta = features_ssta[:,:,:].load()
    
    # masking
    full_mask_land = features_ssta
    full_masked = full_mask_land.where(full_mask_land != 0)
    binary_out_afterlandmask=np.isfinite(full_masked)
    newmask = np.isfinite(ds.SST[0,:,:,:][:])
    
    # blobs
    Tracker = ocetrac.Tracker(binary_out_afterlandmask[:,:,:], 
                              newmask, 
                              radius=radius_val, 
                              min_size_quartile=min_size_quartile_val, 
                              timedim = 'time', 
                              xdim = 'lon', 
                              ydim='lat', 
                              positive=True)
    blobs = Tracker.track()
    blobs.attrs
    mo = Tracker._morphological_operations()
    return detrended, blobs

def merged_fieldlabel_xarray(
    detrended : xr.DataArray, 
    blobs : xr.DataArray) -> xr.DataArray:
    """
    Merges temperature anomalies with the object identifier labels.
    Parameters
    ------------
    detrended : xarray.DataArray (time:1980, lat: 192, lon: 288)
        Consists of SST anomalies.
    blobs : xarray.DataArray 'labels' (time: 1980, lat: 192, lon: 288)
        Consists of object identifier labels.
    Returns
    ------------
    xarray.DataArray with data variables SSTA and labels
    """
    detrended.name = 'SSTA'
    return xr.merge([detrended, blobs])

def calc_number_of_objects(merged_xarray : xr.Dataset) -> int:
    """
    Find the total number of objects in an event file
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    Returns
    ------------
    Integer: total number of objects in an event file
    """
    return len(np.unique(merged_xarray.labels)) - 1

def calc_initialization(
    merged_xarray : xr.Dataset, 
    mhw_id : int):
    """
    Gets the initialization state information
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. Timestep when the object first appears
        2. Image of the object
        3. Month when the object initializes
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    bymonth = np.resize(np.arange(1,13),12*166)[1:-11]
    month = bymonth[first_timestep]
    return first_timestep, one_obj.SSTA[first_timestep,:,:], month

def convert_from_modeltimestep_to_months(time_stamp: int):
    """
    Converts the model timestep to a month in the year
    Parameters
    ------------
    timestep : int 
    Returns
    ------------
    month
    """
    bymonth = np.resize(np.arange(1,13),12*166)[1:-11]
    month = bymonth[first_timestep]
    # todo: check output types
    return month

def _get_labels(binary_images):
    """
    Label binary images at each timestep using skimage.measure.label
    Parameters
    ------------
    binary_images : 
    Returns
    ------------
        1. blobs_labels
    """
    # todo: check input types
    # todo: check output types
    blobs_labels = label_np(binary_images, background=0)
    return blobs_labels

def _get_centroids(sub_labels):
    """
    Find the centroids of objects assigned to each unique label using
    skimage.measure.regionprops. 
    Parameters
    ----------
    sub_labels : 
    Returns
    -------
        1. centroids
    Notes
    -------
    Because of the image transformation, we re-assign longitude coordinates 
    greater than 360 degrees to an appropriate coordinate between 0 and 360°
    """
    # todo: check input types
    # todo: check output types
    props = regionprops(sub_labels.astype('int'))
    centroids = [(float(sub_labels.lat[round(p.centroid[0])].values),
                  float(sub_labels.lon[round(p.centroid[1])].values)) for p in props]
    
    for i in range(0,len(centroids)):
        if centroids[i][1] >= 359.75:
            centroids[i] = (centroids[i][0], list(centroids[i])[1] - 359.75)
    return centroids

def _get_center_of_mass(intensity_image : xr.DataArray):
    """
    Finds the centers of mass of objects assigned to each unique
    label using scipy.ndimage.
    Parameters
    ------------
    intensity_image : xarray.DataAarray
    Returns
    ------------
        1. w_centroids 
    Notes
    ------------
    There can be more than 1 centers of mass per object per timestep.
    """
    # todo: check output types
    img = intensity_image.fillna(0)
    com = ndimage.center_of_mass(img.data)
    # centroid_list.append((float(img.lat[round(ndimage.com[0])].values),float(img.lon[round(ndimage.center_of_mass(img.data)[1])].values)))
    # props = regionprops(sub_labels, intensity, )
    w_centroids = [(float(img.lat[round(com[0])].values),
                  float(img.lon[round(com[1])].values))]
    
    for i in range(0,len(w_centroids)):
        if w_centroids[i][1] >= 359.75:
            w_centroids[i] = (w_centroids[i][0], list(w_centroids[i])[1] - 359.75)
    return w_centroids

def forOneMHW_labels_only(
    merged_xarray : xr.Dataset, 
    mhw_id : int):
    """
    Gets the labels for one object and all its timesteps
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. forOneMHW_only_timesteps
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = calc_duration(merged_xarray, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = one_obj.labels[timesteps_to_choose_from,:,:]
    return forOneMHW_only_timesteps

def forOneMHW_SSTA_only(
    merged_xarray : xr.Dataset, 
    mhw_id : int):
    """
    Gets the sea surface temperature anomalies for one object 
    and all its timesteps
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. forOneMHW_only_timesteps
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = calc_duration(merged_xarray, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = one_obj.SSTA[timesteps_to_choose_from,:,:]
    return forOneMHW_only_timesteps