# import s3fs
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
from typing import Dict, Optional, List, Tuple
from scipy import ndimage

import measures

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
    ----------
    member_id_val : int, default 0
        The integer ID of the member to process
    threshold_val : float, default 0.9
        The threshold percentile (e.g., 0.9) used to identify SST anomalies that exceed the threshold as extreme.
    radius_val : int, default 3
        The number of grid cells that define the width of the structuring element used in morphological operations.
    min_size_quartile_val : float, default 0.75
        The quartile value used to subsample objects based on their distribution of area.
    start_val : int, default 0
        The start index (in months) for the CESM2 LE dataset. Valid range is 0 to 1980 (165 years).
    end_val : int, default 1980
        The end index (in months) for the CESM2 LE dataset. Valid range is 0 to 1980 (165 years).

    Returns
    ----------
    detrended : xarray.DataArray, shape (time: 1980, lat: 192, lon: 288)
        The detrended SST anomalies.
    blobs : xarray.DataArray, shape (time: 1980, lat: 192, lon: 288)
        The labeled features identified by Ocetrac.
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

def combine_sst_and_labels(detrended: xr.DataArray, blobs: xr.DataArray) -> xr.Dataset:
    """
    Merge the given SST anomaly data array with the object identifier labels data array.

    Parameters
    ----------
    detrended : xarray.DataArray of shape (time: 1980, lat: 192, lon: 288)
        A data array containing SST anomalies.
    blobs : xarray.DataArray of shape (time: 1980, lat: 192, lon: 288)
        A data array containing object identifier labels.

    Returns
    -------
    xarray.Dataset
        A dataset containing the following data variables:
        - 'SSTA': SST anomaly data array
        - 'labels': object identifier labels data array
    """
    detrended.name = 'SSTA'
    return xr.merge([detrended, blobs])

def count_objects(merged_xarray: xr.Dataset) -> int:
    """
    Counts the total number of objects in an event file.

    Parameters
    ----------
    merged_xarray : xr.Dataset
        A merged xarray containing the data variables 'SSTA' and 'labels'.

    Returns
    -------
    int
        The total number of objects in the 'labels' data variable, excluding the background.

    Notes
    -----
    The 'labels' data variable should contain integer labels that identify connected regions in the 'SSTA' data variable.
    The value 0 is reserved for the background and is not considered an object.
    """
    return len(np.unique(merged_xarray.labels)) - 1

def get_initial_state(merged_xarray: xr.Dataset, mhw_id: int) -> Tuple[int, xr.DataArray, int]:
    """
    Get initialization state information for a specific object in a merged xarray.

    Parameters
    ----------
    merged_xarray : xarray.Dataset
        A merged xarray containing the data variables 'SSTA' and 'labels'.
    mhw_id : int
        A specific object identifier.

    Returns
    -------
    Tuple[int, xr.DataArray, int]
        A tuple containing the following information about the object:
        1. The timestep when the object first appears in the 'labels' data variable.
        2. The image of the object at the initialization timestep, represented as an xarray DataArray object
           with dimensions ('latitude', 'longitude').
        3. The month (1-12) when the object initializes.

    Notes
    -----
    This function assumes that the first timestep in the model corresponds to the first month of the year.
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    bymonth = np.resize(np.arange(1,13),12*166)[1:-11]
    month = bymonth[first_timestep]
    return first_timestep, month, one_obj.SSTA[first_timestep,:,:]

def convert_timestep_to_month(time_stamp: int) -> int:
    """
    Converts the model timestep to a month in the year.

    Parameters
    ----------
    time_stamp : int
        The model timestep to convert to a month.

    Returns
    -------
    int
        An integer value representing the month (1-12) that corresponds to the input timestep.

    Notes
    -----
    This function assumes that the first timestep in the model corresponds to the first month of the year.
    """
    bymonth = np.resize(np.arange(1, 13), 12 * 166)[1:-11]
    month = bymonth[time_stamp]
    return month

def _get_labels(binary_images: np.ndarray) -> np.ndarray:
    """
    Label binary images at each timestep using skimage.measure.label.

    Parameters
    ----------
    binary_images : numpy.ndarray
        A 3D numpy array of binary images where each element in the array represents a single timestep. Each binary image
        is a 2D array of booleans.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array of labels where each element in the array represents a single timestep. Each label image is a
        2D array of integers where each pixel is assigned a unique identifier.

    Notes
    -----
    The binary_images array should have shape (time, latitude, longitude).
    """
    blobs_labels = label_np(binary_images, background=0)
    return blobs_labels

def _get_centroids(sub_labels: xr.DataArray) -> List[Tuple[float, float]]:
    """
    Find the centroids of objects assigned to each unique label using skimage.measure.regionprops.

    Parameters
    ----------
    sub_labels : xarray.DataArray
        A 2D array of labels where each pixel is assigned a unique identifier. The array should have dimensions 
        ('latitude', 'longitude').

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples, where each tuple contains the latitude and longitude coordinates of the centroid of 
        each object. There may be more than one centroid per object per timestep.

    Notes
    -----
    The input sub_labels should be a 2D array. Because of the image transformation, any longitude coordinates
    greater than 360 degrees will be re-assigned to an appropriate coordinate between 0 and 360°.
    """
    props = regionprops(sub_labels.astype('int'))
    centroids = [(float(sub_labels.lat[round(p.centroid[0])].values),
                  float(sub_labels.lon[round(p.centroid[1])].values)) for p in props]
    
    for i in range(len(centroids)):
        if centroids[i][1] >= 359.75:
            centroids[i] = (centroids[i][0], centroids[i][1] - 360.0)
    return centroids

def _get_center_of_mass(intensity_image: xr.DataArray) -> List[Tuple[float, float]]:
    """
    Find the centers of mass of objects assigned to each unique label in the input intensity image, using the 
    scipy.ndimage module.
    
    Parameters
    ----------
    intensity_image : xarray.DataArray
        An xarray.DataArray object containing the intensity data. This should be a 2D array with dimensions 
        ('latitude', 'longitude').
    
    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples containing the latitude and longitude coordinates of the center of mass
        of each object in the image. There may be more than one center of mass per object per timestep.
    
    Notes
    -----
    Any NaN values in the input image are replaced with 0.0 before computing the center of mass. The resulting
    latitude and longitude values are adjusted so that longitudes greater than or equal to 360 degrees are wrapped
    back to the range [0, 360).
    """
    img = intensity_image.fillna(0)
    com = ndimage.center_of_mass(img.data)

    w_centroids = [(float(img.lat[round(com[0])].values),
                        float(img.lon[round(com[1])].values) % 360) for coms in com]
    for i in range(0,len(w_centroids)):
            if w_centroids[i][1] >= 359.75:
                w_centroids[i] = (w_centroids[i][0], list(w_centroids[i])[1] - 359.75)
    return list(set(w_centroids))

def extract_mhw_labels(merged_xarray: xr.Dataset, mhw_id: int) -> xr.DataArray:
    """
    Extracts labels for a single marine heatwave (MHW) object and all its timesteps.
    
    Parameters
    ----------
    merged_xarray : xarray.Dataset
        A dataset containing the data variables SSTA and labels
    mhw_id : int
        The unique identifier of the MHW object.
        
    Returns
    -------
    xarray.DataArray
        A 3D array containing the labels for the specified MHW object and all its timesteps.
    """
    # Extract the MHW object
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    
    # Find the timesteps where the MHW object is present
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = measures.calc_duration(merged_xarray, mhw_id)
    
    # Choose the timesteps where the MHW object is present and extract the labels
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = one_obj.labels[timesteps_to_choose_from,:,:]
    
    return forOneMHW_only_timesteps

def extract_mhw_ssta(merged_xarray: xr.Dataset, mhw_id: int) -> xr.DataArray:
    """
    Extracts sea surface temperature anomalies (SSTA) for a single marine heatwave (MHW) object and all its timesteps.
    
    Parameters
    ----------
    merged_xarray : xarray.Dataset
        A dataset containing the data variables SSTA and labels
    mhw_id : int
        The unique identifier of the MHW object.
        
    Returns
    -------
    xarray.DataArray
        A 3D array containing the SSTA values for the specified MHW object and all its timesteps.
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(one_obj.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = measures.calc_duration(merged_xarray, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = one_obj.SSTA[timesteps_to_choose_from,:,:]
    return forOneMHW_only_timesteps
