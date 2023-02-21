import numpy as np
import xarray as xr
from skimage import img_as_float
from skimage.morphology import convex_hull_image
from skimage.measure import label as label_np, regionprops, find_contours
from scipy import ndimage
from scipy.interpolate import interp1d
from haversine import haversine, Unit
from typing import Dict, Optional, List, Tuple

import utils

def calc_duration(merged_xarray: xr.Dataset, mhw_id: int) -> int:
    """
    Calculate the duration of an object in the merged_xarray, i.e., the number of timesteps it appears in.

    Parameters
    ----------
    merged_xarray : xr.Dataset
        A dataset containing the object labels and data variables, including SSTA.
    mhw_id : int
        The identifier for the object.

    Returns
    -------
    int
        The number of timesteps where the object with the specified identifier appears in the merged_xarray.
    """
    return len(merged_xarray.where(merged_xarray.labels==mhw_id, drop=True).time)

def calc_cumulative_intensity(merged_xarray: xr.Dataset, mhw_id: int) -> Tuple[float, xr.DataArray]:
    """
    Calculates the cumulative intensity of the object identifier in the merged_xarray,
    (1) as the sum of all its grid points over all timesteps,
    (2) as the sum of the object's intensity at each timestamp.
    
    Parameters
    ----------
    merged_xarray : xr.Dataset
        An xarray dataset with SSTA and labels data variables.
    mhw_id : int
        A specific object identifier.
        
    Returns
    -------
    Tuple[float, xr.DataArray]
        A tuple containing:
        1. cumulative_intensity : float
            The cumulative intensity of the object identifier, calculated as the sum
            of all its grid points over all timesteps.
        2. cumulative_intensity_per_timestamp : xr.DataArray
            The cumulative intensity of the object identifier, calculated as the sum
            of the object's intensity at each timestamp, for all timestamps.
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    cumulative_intensity = one_obj.SSTA.sum()
    cumulative_intensity_per_timestamp = one_obj.SSTA.sum(axis=(1,2))
    return cumulative_intensity, cumulative_intensity_per_timestamp

def calc_mean_intensity(
    merged_xarray: xr.Dataset, 
    mhw_id: int
) -> Tuple[float, xr.DataArray]:
    """
    Calculates the mean intensity of an object as a mean of all its grid points over all timesteps, 
    and the mean intensity of the object at each timestamp.
    
    Parameters
    ----------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
        The input dataset.
    mhw_id : int
        The specific object identifier.

    Returns
    -------
    Tuple[float, xr.DataArray]:
        Returns a tuple with the following elements:
        
        1. mean_intensity : float
            The mean intensity of the object.
        
        2. mean_intensity_per_timestamp : xr.DataArray
            The mean intensity of the object at each timestamp.
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    mean_intensity = one_obj.mean()
    mean_intensity_per_timestamp = one_obj.SSTA.mean(axis=(1,2))
    return mean_intensity.SSTA, mean_intensity_per_timestamp

def calc_maximum_intensity(merged_xarray: xr.Dataset, mhw_id: int) -> Tuple[float, xr.DataArray]:
    """
    Calculate the maximum intensity of the object for each timestep and over all its grid points.
    
    Parameters:
    -----------
    merged_xarray : xarray.Dataset
        An xarray dataset with data variables SSTA and labels
    mhw_id : int
        A specific object identifier
    
    Returns:
    --------
    Tuple of:
        - max_intensity: float
            Maximum intensity of the object over all its grid points and all timesteps
        - max_intensity_per_timestamp: xr.DataArray
            Maximum intensity of the object at each timestamp
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    max_intensity = one_obj.max()
    max_intensity_per_timestamp = one_obj.SSTA.max(axis=(1,2))
    return max_intensity.SSTA, max_intensity_per_timestamp

def calc_std_intensity(merged_xarray: xr.Dataset, mhw_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the standard deviation of the intensity of the object, both as the overall stdev 
    of all its grid points over all timesteps and as a stdev of the object at each timestep.
    
    Parameters
    ----------
    merged_xarray : xarray.Dataset
        A dataset with data variables `SSTA` and `labels`.
    mhw_id : int
        The specific object identifier for which to calculate the standard deviation.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        A tuple containing:
        1. `std_intensity` : np.ndarray
            The overall standard deviation of the object's intensity.
        2. `std_intensity_per_timestamp` : np.ndarray
            The standard deviation of the object's intensity at each timestamp.
    """
    one_obj = merged_xarray.where(merged_xarray.labels == mhw_id, drop=True)
    std_intensity = one_obj.SSTA.std()
    std_intensity_per_timestamp = one_obj.SSTA.std(axis=(1, 2))
    return std_intensity, std_intensity_per_timestamp
    
def calc_spatial_extent(
    merged_xarray: xr.Dataset, 
    mhw_id: int, 
    coords_full: bool = False
):
    """
    Calculates the spatial extent of an object (area)
    
    Parameters
    ----------
    merged_xarray : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    coords_full : bool, optional
        If True, returns the full list of coordinates for each time step, by default False
    
    Returns
    -------
    dict
        Dictionary containing the following keys:
        - spatial_extents: List of areas for each time step
        - max_spatial_extent: Maximum area of the object
        - max_spatial_extent_time: Time step when the object had the maximum area
        - mean_spatial_extent: Mean area of the object
        - cumulative_spatial_extent: Cumulative area of the object
        - coords_full (optional): List of coordinates for each time step
    """
    one_obj = merged_xarray.where(merged_xarray.labels == mhw_id, drop=True)
    
    spatial_extents = []
    coords_full_lst = []
    
    for i in range(len(one_obj.time)):
        for_onetimestep_stacked = one_obj.labels[i,:,:].stack(zipcoords=['lat','lon'])
        intermed = for_onetimestep_stacked[for_onetimestep_stacked.notnull()].zipcoords.values
        coords = list(map(tuple, intermed))
        coords_full_lst.append(coords)
        y, x = zip(*coords)
        # approximate conversions from https://en.wikipedia.org/wiki/Latitude
        dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]
        dlat = (110.574 *1) * np.ones(len(dlon))
        area = np.sum(dlon*dlat)
        spatial_extents.append(area)
    
    max_spatial_extent = max(spatial_extents)
    max_spatial_extent_time = np.argmax(spatial_extents)
    mean_spatial_extent = np.mean(spatial_extents)
    cumulative_spatial_extent = sum(spatial_extents)
    
    results = {
        "spatial_extents": spatial_extents,
        "max_spatial_extent": max_spatial_extent,
        "max_spatial_extent_time": int(max_spatial_extent_time),
        "mean_spatial_extent": mean_spatial_extent,
        "cumulative_spatial_extent": cumulative_spatial_extent,
    }
    
    if coords_full:
        results["coords_full"] = coords_full_lst
    
    return results

def calc_perimeter(merged_xarray: xr.Dataset, mhw_id: int, duration: int) -> list:
    """
    Calculates the perimeter of the object at each timestamp.

    Parameters
    ----------
    merged_xarray : xarray.Dataset
        A dataset containing data variables SSTA and labels.
    mhw_id : int
        The specific object identifier.
    duration : int
        The number of timesteps where the object with the specified 
        identifier appears in the merged_xarray.
    Returns
    -------
    list
        A list of perimeters of the object at each timestamp.
    """
    obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    timesteps = np.arange(utils.calc_initialization(merged_xarray, mhw_id)[:2][0], duration)
    long_range = interp1d([0,360],[-180,180])
    perimeters = []
    
    for i in timesteps:
        bw = obj.labels[i,:,:].values > 0
        contours = find_contours(bw)
        distances = []
        
        for contour in contours:
            lats = obj.lat.values[contour[:,0].astype(int)]
            lons = obj.lon.values[contour[:,1].astype(int)]    
            coords = list(zip(lats, long_range(lons)))

            for ind in range(len(coords)-1):
                distances.append(haversine(coords[ind], coords[ind+1],Unit.KILOMETERS))
            
            distances.append(haversine(coords[-1], coords[0],Unit.KILOMETERS))
        
        perimeters.append(np.sum(distances))
    
    return perimeters

def calc_perimeter_vs_area(
    spatial_extents : list, 
    perimeters : list) -> np.ndarray:
    """
    Calculates the perimeter versus area of an object
    at each timestamp 
    
    Parameters
    ----------
    spatial_extents : list of areas
    perimeters : list of perimeters
    
    Returns
    ----------
    np.ndarray of percentages of perimeter versus spatial extent
    
    Notes
    ----------
    Gives an idea of how deformed an object is.
    Also gives an idea of optimal shapes.
    We have squared units so that we return a unitless number.
    """
    return ((np.asarray(perimeters)**2)/np.asarray(spatial_extents))

def calc_complement_to_deformormation(coords_full: list, spatial_extents: list) -> np.ndarray:
    """
    Calculates the complement to the deformation, which is the fraction of overlapped domain
    between consecutive timestamps.
    
    Parameters
    ----------
    coords_full : list of arrays
        A list of numpy arrays where each array contains the latitude and
        longitude coordinates where the object has an imprint at each timestamp.
    spatial_extents : list of areas
        A list of spatial extents where each element corresponds to the area
        covered by the object at each timestamp.
    
    Returns
    -------
    np.ndarray
        An array of the ratio of shared area between consecutive timestamps.
    
    Notes
    -----
    This function is my implementation of a measure from Sun, D., Jing, Z., Li, F., 
    Wu, L., "Characterizing Global Marine Heatwaves Under a Spatio-temporal Framework," 
    Progress in Oceanography (2022).
    """
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
            sharedarea_ls.append((sharedarea/ (spatial_extents[i] + spatial_extents[i+1])))
        
        else:
            sharedarea_ls.append(0)
            
    return np.array(sharedarea_ls)

def calc_deformation(sharedarea_ls: list) -> np.ndarray:
    """
    Calculates the deformation, which is a measure of the extent to which an 
    object deviates from a smooth, circular shape. The deformation is defined 
    as the fraction of non-overlapped domain occupied by the object at 2 
    different times.
    
    Parameters
    ----------
    sharedarea_ls : list
        A list of the ratio of shared area between consecutive timestamps.
    
    Returns
    -------
    np.ndarray
        An array of the deformation of the object at each timestamp.
    
    Notes
    -----
    This function is my implementation of a measure that calculates the deformation,
    which is the complement to the overlap or shared area between consecutive timestamps.
    The calculation is based on the algorithm presented in Sun, D., Jing, Z., Li, F., 
    Wu, L., "Characterizing Global Marine Heatwaves Under a Spatio-temporal Framework," 
    Progress in Oceanography (2022).
    """
    return np.asarray(1 - np.asarray(sharedarea_ls))

def calc_largest_and_smallest_timestamp(spatial_extents: List[float]) -> Tuple[float, float]:
    """
    Calculates the timestamp at which the object has the largest and smallest spatial extents.
    
    Parameters
    ----------
    spatial_extents : list of floats
        The spatial extents of the object at each timestamp.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the percentage of the time series at which the object has its 
        largest and smallest spatial extents, respectively.
    """
    when_large = np.argmax(spatial_extents) 
    when_small = np.argmin(spatial_extents)
    return when_large, when_small

def calc_spatial_cross_correlation(merged_xarray: xr.Dataset, mhw_id: int, duration: int) -> np.ndarray:
    """
    Calculates the spatial cross correlation of an object in the merged_xarray dataset

    Parameters
    ----------
    merged_xarray : xr.Dataset
        A dataset containing the merged Sea Surface Temperature Anomaly (SSTA) data and labels
    mhw_id : int
        The identifier of the object to compute the spatial cross correlation for
    duration : int
        The number of timesteps where the object with the specified 
        identifier appears in the merged_xarray.
    Returns
    -------
    np.ndarray
        An array containing the spatial cross correlation values for the object at each timestep
    """
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=False)
    first_timestep, first_array, month = utils.calc_initialization(merged_xarray, mhw_id)
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

def calc_ratio_convexhullarea_vs_objarea(merged_xarray: xr.Dataset, mhw_id: int) -> list:
    """
    Calculate the ratio of the area of an object to the area of its convex hull.

    Parameters
    ----------
    merged_xarray : xr.Dataset
        A dataset containing the merged Sea Surface Temperature Anomaly (SSTA) data and labels
    mhw_id : int
        The identifier of the object to compute the ratio for

    Returns
    -------
    list
        An list representing the ratio of the area of the object to the area of its convex hull.
        The length of the array is equal to the number of timesteps for which the object is present in the dataset.
    """
    one_obj = merged_xarray.labels.where(merged_xarray.labels==mhw_id)
    imoverchull = []
    for i in range(len(one_obj.time)):
        if np.nansum(one_obj[i].values) > 0:
            numer = np.sum(img_as_float(convex_hull_image(one_obj[i].values == mhw_id)))
            denom = np.sum(img_as_float(one_obj[i].values == mhw_id))
            imoverchull.append(numer/denom)
    return imoverchull

def calc_centroids_per_timestep(extract_mhw_labels: xr.DataArray, timestep: int) -> list:
    """
    Finds the locations of the centroids of an object at each timestep.

    Parameters
    ----------
    extract_mhw_labels : xarray.DataArray
        A 3D array representing labels for each timestep of a single MHW object. The labels array is obtained using
        the `forOneMHW_labels_only` function in utils.py.
    timestep : int
        The index of the timestep for which centroids are to be computed.

    Returns
    -------
    list
        A list of (latitude, longitude) tuples representing the centroids of the MHW object at the specified timestep.

    Notes
    -----
    - There can be more than 1 centroid per object per timestep.
    - `forOneMHW_labels_only` function in utils.py should be run before calling this function.
    """
    sub_labels = extract_mhw_labels.isel(time=timestep)
    edge_labels = np.unique(np.concatenate((sub_labels[:,0].values, sub_labels[:,-1].values)))
    nonedge_labels = np.setdiff1d(np.unique(sub_labels.values), edge_labels)

    centroid_list = []
    for label in nonedge_labels:
        mask = (sub_labels == label)
        centroid_list.append(utils._get_centroids(mask))

    for left_label, right_label in zip(edge_labels[:-1], edge_labels[1:]):
        left_mask = (sub_labels == left_label)
        right_mask = (sub_labels == right_label)
        lon_edge = sub_labels.coords['lon'][0].item()

        # Shift left mask to the right and join with right mask
        shifted_left_mask = left_mask.roll(lon=-1)
        shifted_left_mask.coords['lon'] = (shifted_left_mask.coords['lon'] + 360) 
        joined_mask = shifted_left_mask.combine_first(right_mask)

        east_mask = joined_mask.where(joined_mask.lon > lon_edge, drop=True)
        west_mask = joined_mask.where(joined_mask.lon <= lon_edge, drop=True)
        new_mask = xr.concat([east_mask, west_mask], dim="lon")

        centroid_list.append(utils._get_centroids(new_mask))

    return list(set([item for sublist in centroid_list for item in sublist]))

def calc_com_per_timestep(extract_mhw_labels: xr.DataArray, 
                          extract_mhw_ssta: xr.DataArray, 
                          timestep: int) -> List[Tuple[float, float]]:
    """
    Calculates the center of mass of each object in the input data at the specified timestep.

    Parameters
    ----------
    extract_mhw_labels : xarray.DataArray
        Array of labels for each object at each timestep.
    extract_mhw_ssta : xarray.DataArray
        Array of SSTA (sea surface temperature anomaly) values for each object at each timestep.
    timestep : int
        Index of the timestep for which to calculate the center of mass.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (lat, lon) tuples representing the center of mass of each object at the specified timestep.

    Notes
    -----
    - This function should be called after running the 'forOneMHW_labels_only' and 'forOneMHW_SSTA_only'
      functions in the 'utils' module.
    """

    timestep_of_interest = extract_mhw_labels[timestep,:,:]
    SSTA_in_timestep = extract_mhw_ssta[timestep,:,:]
    sub_labels = xr.DataArray(
        utils._get_labels(timestep_of_interest), 
        dims=timestep_of_interest.dims, 
        coords=timestep_of_interest.coords).where(timestep_of_interest>0, 
                                                  drop=False, 
                                                  other=np.nan)
    com_list = []

    for i in np.unique(sub_labels.values.flatten()[~np.isnan(sub_labels.values.flatten())]):

        if i not in np.unique(sub_labels[:,-1:]) and i not in np.unique(sub_labels[:,:1]):
            sub_labels_i = sub_labels.where(sub_labels==i, drop=False, other=np.nan)
            sub_SSTAs_i = SSTA_in_timestep.where(sub_labels==i, drop=False, other=np.nan)
            intensity_image = sub_SSTAs_i
            com_list.append(utils._get_center_of_mass(intensity_image)[0])

        else:
            sub_labels_i = sub_labels.where(sub_labels==i, drop=False, other=np.nan)
            sub_SSTAs_i = SSTA_in_timestep.where(sub_labels==i, drop=False, other=np.nan)
            lon_edge = sub_labels_i[:,-1:].lon.item() if i in np.unique(sub_labels[:,-1:]) else sub_labels_i[:,:1].lon.item()

            if lon_edge < 358.75:
                sub_labels_left = sub_labels.where(sub_labels==i, drop=True)
                SSTA_left = SSTA_in_timestep.where(SSTA_in_timestep.lon <= lon_edge, drop=True)
                # sub_labels_right = sub_labels.where(sub_labels!=i, drop=False, other=np.nan).where(sub_labels.right > lon_edge, drop=True)
                SSTA_right = SSTA_in_timestep.where(SSTA_in_timestep.lon > lon_edge, drop=True)
                append_east = xr.concat([sub_labels_left.where(sub_labels_left.lon >= lon_edge, drop=True), sub_labels_right], dim="lon")
                append_east_SSTA = xr.concat([SSTA_left.where(SSTA_left.lon >= lon_edge, drop=True), SSTA_right], dim="lon")
                intensity_image = append_east_SSTA
                com_list.append(utils._get_center_of_mass(append_east_SSTA.where(intensity_image)[0])[0])
    return com_list

def calc_displacement(
    labels: xr.DataArray, 
    ssta: xr.DataArray,
    lat_dim: Optional[str] = 'lat',
    lon_dim: Optional[str] = 'lon'
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Tracks the centroid and center of mass displacement for a set of labels and SST anomalies.

    Parameters
    ----------
    labels : xarray.DataArray
        A 2D or 3D array with labeled regions of interest, where the labels correspond to MHW events.
    ssta : xarray.DataArray
        A 2D or 3D array with sea surface temperature anomalies that correspond to the same time and space as labels.
    lat_dim : str, optional
        The name of the latitude dimension in the input arrays, by default 'lat'.
    lon_dim : str, optional
        The name of the longitude dimension in the input arrays, by default 'lon'.
    
    Returns
    ----------
    A dictionary with the following keys:
        - 'centroid_coords': a list of tuples with the centroid coordinates for each timestep.
        - 'com_coords': a list of tuples with the center of mass coordinates for each timestep.
        - 'centroid_displacements': a list of distances between centroid coordinates for consecutive timesteps.
        - 'com_displacements': a list of distances between center of mass coordinates for consecutive timesteps.
        - 'centroid_xrcoords': a list of xarray coordinates for the centroid for each timestep.
        - 'com_xrcoords': a list of xarray coordinates for the center of mass for each timestep.

    Notes
    ----------
    This function tracks the centroid and center of mass displacement (one per timestep) for a set of labels and 
    SST anomalies. It can handle 2D or 3D input arrays, where the first dimension is assumed to be time. The 
    displacement is calculated using the haversine formula, assuming a spherical Earth. The coordinates are 
    converted from the input latitude and longitude ranges to (-90, 90) for latitude and (-180, 180) for longitude.
    """

    centroid_list = []; centroid_xrcoords_ls = []
    center_of_mass_list = []; com_xrcoords_ls = []
    for i in range(labels.shape[0]):
        labels = xr.where(labels > 0, 1, np.nan)
        ssta = xr.where(ssta > 0, 1, np.nan)
        img_cent_xr_coords = utils._get_center_of_mass(labels[i,:,:])
        centroid_xrcoords_ls.append(img_cent_xr_coords[0])
        img_SSTA_xr_coords = utils._get_center_of_mass(ssta[i,:,:])
        com_xrcoords_ls.append(img_SSTA_xr_coords[0])
        img_cent = labels[i,:,:].fillna(0)
        img_SSTA = ssta[i,:,:].fillna(0)
        centroid_list.append(ndimage.center_of_mass(img_cent.data))
        center_of_mass_list.append(ndimage.center_of_mass(img_SSTA.data))

    y_val_cent = list(zip(*centroid_list))[0]; x_val_cent = list(zip(*centroid_list))[1]
    y_val_com = list(zip(*center_of_mass_list))[0]; x_val_com = list(zip(*center_of_mass_list))[1]

    convert_long_range = interp1d([0,360],[-180,180]); convert_lat_range = interp1d([0,180],[-90,90])

    coords_cent = list(zip(convert_lat_range(x_val_cent), convert_long_range(y_val_cent)))
    coords_com = list(zip(convert_lat_range(x_val_com), convert_long_range(y_val_com)))

    distance_cent_ls = []; distance_com_ls = []
    for i in range(len(coords_cent)-1):
        distance_cent = haversine(coords_cent[i], coords_cent[i+1],Unit.KILOMETERS)
        distance_cent_ls.append(distance_cent)
        distance_com = haversine(coords_com[i], coords_com[i+1],Unit.KILOMETERS)
        distance_com_ls.append(distance_com)
        
    results = {
        
        "centroid_coords": centroid_list,
        "com_coords": center_of_mass_list,
        "centroid_displacements": distance_cent_ls,
        "com_displacements": distance_com_ls,
        "centroid_xrcoords": centroid_xrcoords_ls,
        "com_xrcoords": com_xrcoords_ls,
    }
    
    return results