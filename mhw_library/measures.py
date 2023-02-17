import numpy as np
import xarray as xr
from skimage import img_as_float
from skimage.morphology import convex_hull_image
from skimage.measure import label as label_np, regionprops, find_contours
from scipy import ndimage
from scipy.interpolate import interp1d
from haversine import haversine, Unit
from typing import Tuple

def calc_duration(
    merged_xarray : xr.Dataset, 
    mhw_id : int) -> int:
    """
    Calculates the number of timestamps the object identifier
    appears in the merged_xarray
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
    Integer: total number of timesteps where the object identifier appears
    """
    return len(merged_xarray.where(merged_xarray.labels==mhw_id, drop=True).time)

def calc_cumulative_intensity(
    merged_xarray : xr.Dataset, 
    mhw_id : int):
    """
    Calculates
    (1) cumulative intensity of the object as
    a sum of all its grid points over all timesteps
    (2) cumulative intensity of the object at each timestamp
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. cumulative_intensity 
        2. cumulative_intensity_per_timestamp  
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    cumulative_intensity = one_obj.SSTA.sum()
    cumulative_intensity_per_timestamp = one_obj.SSTA.sum(axis=(1,2))
    # todo: check output types
    return cumulative_intensity, cumulative_intensity_per_timestamp

def calc_mean_intensity(
    merged_xarray : xr.Dataset 
    mhw_id : xr.Dataset):
    """
    Calculates 
    (1) mean intensity of the object as a 
    mean of all its grid points over all timesteps 
    (2) mean intensity of the object at each timestamp
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. mean_intensity 
        2. mean_intensity_per_timestamp  
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    mean_intensity = one_obj.mean()
    mean_intensity_per_timestamp = one_obj.SSTA.mean(axis=(1,2))
    # todo: check output types
    return mean_intensity.SSTA, mean_intensity_per_timestamp

def calc_maximum_intensity(
    merged_xarray : xr.Dataset, 
    mhw_id : xr.Dataset):
    """
    Calculates
    (1) maximum intensity of the object as a
    maximum of all its grid points over all timesteps
    (2) maximum intensity of the object at each timestamp
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. max_intensity
        2. max_intensity_per_timestamp
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    max_intensity = one_obj.max()
    max_intensity_per_timestamp = one_obj.SSTA.max(axis=(1,2))
    # todo: check output types
    return max_intensity.SSTA, max_intensity_per_timestamp

def calc_std_intensity(
    merged_xarray : xr.Dataset,
    mhw_id : int):
    """
    Calculates the (1) standard deviation (stdev) of the intensity of the
    object as a stdev of all its grid points over all timesteps and (2)
    as a stdev of the object at each timestep
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. std_intensity
        2. std_intensity_per_timestamp
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    std_intensity = one_obj.std()
    std_intensity_per_timestamp = one_obj.SSTA.std(axis=(1,2))
    # todo: check output types
    return std_intensity.SSTA, std_intensity_per_timestamp

def calc_spatial_extent(
    merged_xarray : xr.Dataset, 
    mhw_id : int, 
    coords_full : False):
    """
    Calculates the spatial extent of an object (area)
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. coords_full : (optional) coordinates where the object has an imprint
        2. spatial_extents 
        3. max_spatial_extent
        4. max_spatial_extent_time
        5. mean_spatial_extent
        6. cumulative_spatial_extent
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    
    spatial_extents = []
    coords_full = []
    
    # todo: wrap some of the steps below in a function
    for i in range(len(one_obj.time)):
        for_onetimestep_stacked = one_obj.labels[i,:,:].stack(zipcoords=['lat','lon'])
        intermed = for_onetimestep_stacked[for_onetimestep_stacked.notnull()].zipcoords.values
        lats = [x[0] for x in intermed]
        lons = [x[1] for x in intermed]
        coords = list(zip(lats, lons))
        coords_full.append(coords)
        y,x=zip(*coords)
        # todo: reference where the numbers for dlon and dlat
        dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]
        dlat = (110.574 *1) * np.ones(len(dlon))
        area = np.sum(dlon*dlat)
        spatial_extents.append(area)
    
    max_spatial_extent = np.max(spatial_extents)
    max_spatial_extent_time = np.argmax(spatial_extents)
    mean_spatial_extent = np.mean(spatial_extents)
    cumulative_spatial_extent = np.sum(spatial_extents)
    
    # todo: return as dictionary
    # todo: check output types
    if return_bar:
        return coords_full, spatial_extents, max_spatial_extent, max_spatial_extent_time, mean_spatial_extent, cumulative_spatial_extent
    
    else:
        return spatial_extents, max_spatial_extent, max_spatial_extent_time, mean_spatial_extent, cumulative_spatial_extent
    
def calc_perimeter(
    merged_xarray : xr.Dataset, 
    mhw_id : int) -> list:
    """
    Calculates the perimeter of the object at each timestamp
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. perimeter_ls
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    # todo: calc_initialization comes from utils.py, import utils.py correctly
    first_timestep, first_array, month = calc_initialization(merged_xarray, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)

    convert_long_range = interp1d([0,360],[-180,180])
    perimeter_ls = []
    
    for i in timesteps_to_choose_from:
        bw = one_obj.labels[i,:,:].values > 0
        contours = find_contours(bw)
        distance_ls = []
        
        for contour_num in range(len(contours)):
            latitudes = one_obj.lat.values[contours[contour_num][:,0].astype(int)]
            longitudes = one_obj.lon.values[contours[contour_num][:,1].astype(int)]    
            coords = list(zip(latitudes, convert_long_range(longitudes)))

            for ind in range(len(coords)-1):
                distance = haversine(coords[ind], coords[ind+1],Unit.KILOMETERS)
                distance_ls.append(distance)
            
            distance_ls.append(haversine(coords[len(coords)-1], coords[0],Unit.KILOMETERS))
        
        perimeter = np.sum(distance_ls)
        perimeter_ls.append(perimeter)
    
    return perimeter_ls  

def calc_perimeter_vs_area(
    spatial_extents : list, 
    perimeters : list):
    """
    Calculates the perimeter versus area of an object
    at each timestamp 
    Parameters
    ------------
    spatial_extents : list of areas
    perimeters : list of perimeters
    Returns
    ------------
    np.ndarray of percentages of perimeter versus spatial extent
    Notes
    ------------
    Gives an idea of how deformed an object is.
    Also gives an idea of optimal shapes.
    We have sqaured units so that we return a unitless number.
    """
    # todo: check output types
    return ((np.asarray(perimeters)**2)/np.asarray(spatial_extents))

def calc_complement_to_deformormation(
    coords_full, 
    spatial_extents):
    """
    Calculates the fraction of overlapped domain, which is also
    the complement to the deformation
    Parameters
    ------------
    coords_full : coordinates where the object has an imprint
    spatial_extents : areas
    Returns
    ------------
        1. perc_sharedarea_ls
    Notes
    ------------
    This is my implementation of a measure from Sun, D., Jing, Z., Li, F., 
    Wu, L., Characterizing Global Marine Heatwaves Under a Spatio-temporal 
    Framework, Progress in Oceanography (2022), 
    doi: https://doi.org/10.1016/j.pocean. 2022.102947.
    """
    # todo: check input types
    sharedarea_ls = []
    
    for i in range(len(coords_full)-1):
        a_set = set(coords_full[i])
        b_set = set(coords_full[i+1])
        
        if a_set & b_set:
            coords = a_set & b_set
            y,x=zip(*coords)
            dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]
            dlat = (110.574 *1) * np.ones(len(dlon))
            sharedarea = np.sum(dlon*dlat)
            perc_sharedarea_ls.append((sharedarea/ (spatial_extents[i] + spatial_extents[i+1])))
        
        else:
            sharedareaarea = 0
            sharedarea_ls.append((sharedarea/ (spatial_extents[i] + spatial_extents[i+1])))
    # todo: check input types
    return sharedarea_ls

def calc_deformormation(
    sharedarea_ls : list):
    """
    Calculates the deformation, which is the fraction of non-overlapped 
    domain occupied by an object at 2 different times
    Parameters
    ------------
    perc_sharedarea_ls : list
    Returns
    ------------
    np.ndarray : fraction of non-overlapped domain occupied by an object at
    each timestamp
    Notes
    ------------
    This is my implementation of a measure from Sun, D., Jing, Z., Li, F., 
    Wu, L., Characterizing Global Marine Heatwaves Under a Spatio-temporal 
    Framework, Progress in Oceanography (2022), 
    doi: https://doi.org/10.1016/j.pocean. 2022.102947.
    """
    # todo: check output types
    # todo: verify that this is what we want to calculate
    return np.asarray(1 - np.asarray(sharedarea_ls))

def calc_when_large_and_small_timestamp(spatial_extents):
    """
    Finds the timestamp with 
    (1) the largest spatial extent
    (2) the smallest spatial extent
    Parameters
    ------------
    spatial_extents : object areas at each timestep
    Returns
    ------------
        1. when_large
        2. when_small 
    """
    # todo: check input types
    # todo: check output types
    when_large = (np.argmax(spatial_extents) / len(spatial_extents))*100
    when_small = (np.argmin(spatial_extents) / len(spatial_extents))*100
    return when_large, when_small

def calc_spatial_cross_correlation(
    merged_xarray : xr.Dataset, 
    mhw_id : int):
    """
    Calculates the spatial cross correlation of an object
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. cc_image_array
    """
    # todo: check output type
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    first_timestep, first_array, month = initialization(merged_xarray, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    cc_image_array = np.zeros((len(timesteps_to_choose_from), 192,288))    
    
    for i in range(len(timesteps_to_choose_from[:-1])):
        image = one_obj.SSTA[timesteps_to_choose_from[i],:,:].values
        image = np.nan_to_num(image)
        offset_image = one_obj.SSTA[timesteps_to_choose_from[i+1],:,:].values
        offset_image = np.nan_to_num(offset_image)
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        cc_image_array[i,:,:] = np.real(cc_image)
    
    return cc_image_array

def calc_ratio_convexhullarea_vs_objarea(
    merged_xarray : xr.Dataset, 
    mhw_id : int) -> list:
    """
    Calculates the ratio of the convex hull area and the object area
    Parameters
    ------------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. ratio_imoverchull_ls
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    ratio_imoverchull_ls = []
    
    for i in range(len(one_obj.time)):
        image = one_obj.labels[i].values
        image = [image == mhw_id][0]
        chull = convex_hull_image(image)
        chull_asflt = img_as_float(chull.copy())
        image_asflt = img_as_float(image.copy())
        ratio_imoverchull = np.sum(image_asflt)/np.sum(chull_asflt)
        ratio_imoverchull_ls.append(ratio_imoverchull)
    
    return ratio_imoverchull_ls

def calc_centroids_per_timestep(
    forOneMHW_onlylabels_timesteps : xr.DataArray, 
    timestep : int) -> list:
    """
    Finds the locations of the centroids of an object at each timestep
    Parameters
    ------------
    forOneMHW_onlylabels_timesteps : xarray.DataArray. 
    timestep : int
    Returns
    ------------
        1. flat_centroid_list
    Notes
    ------------
    - There can be more than 1 centroid per object per timestep.
    - Need to run forOneMHW_labels_only first (function in utils.py)
    """
    # Step 1. We start with one timestep and get all the sublabels
    timestep_of_interest = forOneMHW_onlylabels_timesteps[timestep,:,:]
    get_sub_lbs = timestep_of_interest
    # todo: the function _get_labels comes from utils.py. Load it correctly.
    sub_labels = _get_labels(get_sub_lbs)
    sub_labels = xr.DataArray(sub_labels, dims=get_sub_lbs.dims, coords=get_sub_lbs.coords)
    sub_labels = sub_labels.where(timestep_of_interest>0, drop=False, other=np.nan)

    # Step 2. We get all the labels on the edges
    edge_right_sub_labels_ = sub_labels[:,-1:]
    edge_right_sub_labels = np.unique(np.unique(edge_right_sub_labels_)[~np.isnan(np.unique(edge_right_sub_labels_))])
    edge_left_sub_labels_ = sub_labels[:,:1]
    edge_left_sub_labels = np.unique(np.unique(edge_left_sub_labels_)[~np.isnan(np.unique(edge_left_sub_labels_))])
    
    edge_labels = np.unique(np.concatenate((edge_right_sub_labels, edge_left_sub_labels)))
    nonedgecases = np.setdiff1d(np.unique(sub_labels), edge_labels)
    nonedgecases = np.unique(nonedgecases[~np.isnan(nonedgecases)])

    centroid_list = []
    for i in nonedgecases:
        sub_labels_nonedgecases = sub_labels.where(sub_labels==i, drop=False, other=np.nan)
        # todo: the function _get_centroids comes from utils.py. Load it correctly.
        centroid_list.append(_get_centroids(sub_labels_nonedgecases))
    
    for i in edge_left_sub_labels:
        sub_labels_left = sub_labels.where(sub_labels==i, drop=True)
        lon_edge = sub_labels_left[:,-1:].lon.item()
        sub_labels_left.coords['lon'] = (sub_labels_left.coords['lon'] + 360) 
        
        for j in edge_right_sub_labels:
            sub_labels_right = sub_labels.where(sub_labels==j, drop=False, other=np.nan)
            east = sub_labels_right.where(sub_labels_right.lon > lon_edge, drop=True)
            append_east = xr.concat([east.where(east.lon >= lon_edge, drop=True), sub_labels_left], dim="lon")
            append_east_binarized = xr.where(append_east > 0, 1, np.nan)
            # todo: the function _get_labels comes from utils.py. Load it correctly.
            sub_labels = _get_labels(append_east_binarized)
            sub_labels = xr.DataArray(sub_labels, dims=append_east_binarized.dims, coords=append_east_binarized.coords)
            sub_labels = sub_labels.where(append_east_binarized>0, drop=False, other=np.nan)
            centroid_list.append(_get_centroids(sub_labels))
    
    flat_centroid_list = list(set([item for sublist in centroid_list for item in sublist]))
    return flat_centroid_list

def calc_com_per_timestep(
    forOneMHW_onlylabels_timesteps : xr.DataArray, 
    forOneMHW_onlySSTA_timesteps : xr.DataArray, 
    timestep : int):
    """
    Finds the center of mass of an object at each timestep. 
    There can only be 1 center of mass per object per timestep.
    Parameters
    ------------
    forOneMHW_onlylabels_timesteps : xarray.DataAarray
    forOneMHW_onlySSTA_timesteps : xarray.DataArray
    timestep : int
    Returns
    ------------
        1. centroid_list
    Notes
    -----------
    - Need to run forOneMHW_labels_only first (function in utils.py)
    - Need to run forOneMHW_SSTA_only first (function in utils.py)
    """
    timestep_of_interest = forOneMHW_onlylabels_timesteps[timestep,:,:] # labels in one given timestep
    SSTA_in_timestep = forOneMHW_onlySSTA_timesteps[timestep,:,:] # SSTA in one given timestep
    # todo: the function _get_labels comes from utils.py. Load it correctly.
    sub_labels = _get_labels(timestep_of_interest) # use skimage to get sub_labels
    sub_labels = xr.DataArray(sub_labels, dims=timestep_of_interest.dims, coords=timestep_of_interest.coords)
    sub_labels = sub_labels.where(timestep_of_interest>0, drop=False, other=np.nan)
    
    edge_right_sub_labels = np.unique(np.unique(sub_labels[:,-1:])[~np.isnan(np.unique(sub_labels[:,-1:]))])
    edge_left_sub_labels = np.unique(np.unique(sub_labels[:,:1])[~np.isnan(np.unique(sub_labels[:,:1]))])
    edge_labels = np.unique(np.concatenate((edge_right_sub_labels, edge_left_sub_labels)))
    nonedgecases = np.setdiff1d(np.unique(sub_labels), edge_labels)
    nonedgecases = np.unique(nonedgecases[~np.isnan(nonedgecases)])
    
    centroid_list = []
    for i in nonedgecases:
        sub_labels_nonedgecases = xr.where(sub_labels==i, SSTA_in_timestep, np.nan)
        sub_labels_nonedgecases_labels = sub_labels.where(sub_labels==i, drop=False, other=np.nan)
        # todo: the function _get_center_of_mass comes from utils.py. Load it correctly.
        centroid_list.append(_get_center_of_mass(sub_labels_nonedgecases)[0])
    
    for i in edge_left_sub_labels:
        sub_labels_left = sub_labels.where(sub_labels==i, drop=True)
        lon_edge = sub_labels_left[:,-1:].lon.item()
        
        if lon_edge < 358.75:
            SSTA_left = SSTA_in_timestep.where((SSTA_in_timestep.lon <= lon_edge), drop=True)
            sub_labels_left.coords['lon'] = (sub_labels_left.coords['lon'] + 360) 
            SSTA_left.coords['lon'] = (SSTA_left.coords['lon'] + 360) 
            
            for j in edge_right_sub_labels:
                sub_labels_right = sub_labels.where(sub_labels==j, drop=False, other=np.nan)
                sub_SSTAs_right = SSTA_in_timestep.where(sub_labels==j, drop=False, other=np.nan)
                east = sub_labels_right.where(sub_labels_right.lon > lon_edge, drop=True)
                east_SSTA = sub_SSTAs_right.where(sub_SSTAs_right.lon > lon_edge, drop=True)
                append_east = xr.concat([east.where(east.lon >= lon_edge, drop=True), sub_labels_left], dim="lon")
                append_east_SSTA = xr.concat([east_SSTA.where(east_SSTA.lon >= lon_edge, drop=True), SSTA_left], dim="lon")
                append_east = xr.where(append_east > 0, 1.0, np.nan)
                centroid_list.append(_get_center_of_mass(append_east_SSTA)[0])
    
    return centroid_list

def calc_displacement(
    forOneMHW_onlylabels_timesteps : xr.DataArray, 
    forOneMHW_onlySSTA_timesteps : xr.DataArray):
    """
    Tracks the centoid and center of masses 
    Parameters
    ------------
    forOneMHW_onlylabels_timesteps : xarray.DataArray
    forOneMHW_onlySSTA_timesteps : xarray.DataArray
    Returns
    ------------
        1. centroid_list
        2. center_of_mass_list
        3. distance_cent_ls
        4. distance_com_ls
        5. centroid_xrcoords_ls
        6. com_xrcoords_ls
    Notes
    ------------
    This function will only work when we have more than 1 timestep for the object and calculates 
    the centroid and center of mass displacement (one per timestep) unlike com_per_timestep
    and centroids_per_timestep. This function should give information about movement (zonal 
    versus meridional, eastward versus westward).
    """
    centroid_list = []; centroid_xrcoords_ls = []
    center_of_mass_list = []; com_xrcoords_ls = []
    for i in range(forOneMHW_onlylabels_timesteps.shape[0]):
        forOneMHW_onlylabels_timesteps = xr.where(forOneMHW_onlylabels_timesteps > 0, 1, np.nan)
        forOneMHW_onlySSTA_timesteps = xr.where(forOneMHW_onlySSTA_timesteps > 0, 1, np.nan)
        # todo: the function _get_center_of_mass comes from utils.py. Load it correctly.
        img_cent_xr_coords = _get_center_of_mass(forOneMHW_onlylabels_timesteps[i,:,:])
        centroid_xrcoords_ls.append(img_cent_xr_coords[0])
        img_SSTA_xr_coords = _get_center_of_mass(forOneMHW_onlySSTA_timesteps[i,:,:])
        com_xrcoords_ls.append(img_SSTA_xr_coords[0])
        img_cent = forOneMHW_onlylabels_timesteps[i,:,:].fillna(0)
        img_SSTA = forOneMHW_onlySSTA_timesteps[i,:,:].fillna(0)
        centroid_list.append(ndimage.center_of_mass(img_cent.data))
        center_of_mass_list.append(ndimage.center_of_mass(img_SSTA.data))

    y_val_cent = list(zip(*centroid_list))[0]
    x_val_cent = list(zip(*centroid_list))[1]
    y_val_com = list(zip(*center_of_mass_list))[0]
    x_val_com = list(zip(*center_of_mass_list))[1]

    convert_long_range = interp1d([0,360],[-180,180])
    convert_lat_range = interp1d([0,180],[-90,90])

    coords_cent = list(zip(convert_lat_range(x_val_cent), convert_long_range(y_val_cent)))
    coords_com = list(zip(convert_lat_range(x_val_com), convert_long_range(y_val_com)))

    distance_cent_ls = []; distance_com_ls = []
    for i in range(len(coords_cent)-1):
        distance_cent = haversine(coords_cent[i], coords_cent[i+1],Unit.KILOMETERS)
        distance_cent_ls.append(distance_cent)
        distance_com = haversine(coords_com[i], coords_com[i+1],Unit.KILOMETERS)
        distance_com_ls.append(distance_com)
    # todo: return dictionary
    # todo: check which are not necessary and make them optional returns
    return centroid_list, center_of_mass_list, distance_cent_ls, distance_com_ls, centroid_xrcoords_ls, com_xrcoords_ls