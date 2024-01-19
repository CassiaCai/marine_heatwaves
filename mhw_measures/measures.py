"""This module contains functions for calculating measures for each mhw."""

import xarray as xr
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage import img_as_float
from skimage.morphology import convex_hull_image
from skimage.measure import label as label_np, regionprops, find_contours
from haversine import haversine, Unit
from typing import Dict, Optional, List, Tuple
import numpy as np

def calc_spatialextent(
    merged_file_by_mhwid: xr.Dataset
):
    """
    Calculates the spatial extent of an object (area) in km2
    
    Parameters:
        merged_file_by_mhwid (xr.Dataset): an xarray.Dataset with data variables SSTA and mhw labels
    
    Returns:
        coords_full:
        spatial_extents:
        max_spatial_extent:
        max_spatial_extent_time:
        mean_spatial_extent:
        cumulative_spatial_extent:
    
    Example usage:
    
    """
    one_mhw = merged_file_by_mhwid.mhw_number[:,:,:]
    for_one_mhw = one_mhw.where(one_mhw > 0, drop=True)
    
    spatial_extents = []
    coords_full = []
    
    for i in range(len(for_one_mhw.time)):
        for_onetimestep_stacked = for_one_mhw[i,:,:].stack(zipcoords=['lat','lon'])
        intermed = for_onetimestep_stacked[for_onetimestep_stacked.notnull()].zipcoords.values
        lats = [x[0] for x in intermed]; lons = [x[1] for x in intermed]
        coords = list(zip(lats, lons))
        coords_full.append(coords)
        y,x=zip(*coords)
        dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]
        dlat = (110.574 *1) * np.ones(len(dlon))
        area = np.sum(dlon*dlat)
        spatial_extents.append(area)
    
    max_spatial_extent = np.max(spatial_extents)
    max_spatial_extent_time = np.argmax(spatial_extents)
    mean_spatial_extent = np.mean(spatial_extents)
    cumulative_spatial_extent = np.sum(spatial_extents)
    
    return coords_full, spatial_extents, max_spatial_extent, max_spatial_extent_time, mean_spatial_extent, cumulative_spatial_extent

def calc_perimeter(merged_file_by_mhwid, timesteps):
    obj = merged_file_by_mhwid
    
    long_range = interp1d([0,360],[-180,180])
    
    perimeters = []

    for i in timesteps:
        bw = obj.mhw_number[i,:,:].values > 0
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

def calc_complement_to_deformormation(coords_full: list, spatial_extents: list) -> np.ndarray:
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
    return np.asarray(1 - np.asarray(sharedarea_ls))

def calc_ratio_convexhullarea_vs_area(merged_file_by_mhwid, duration):
    imoverchull = []
    for i in range(duration):
        one_obj = merged_file_by_mhwid.mhw_number.isel(time=i)
        numer = np.sum(xr.where(one_obj > 0, 1, 0))
        denom = np.sum(img_as_float(convex_hull_image(one_obj>0)))
        frac = numer/denom
        imoverchull.append(frac.values.item())
    return imoverchull

def calc_spatial_cross_autocorrelation(merged_file_by_mhwid, initial_timestep):
    one_obj = merged_file_by_mhwid
    timesteps_to_choose_from = initial_timestep
    sa_image_array = np.zeros((len(timesteps_to_choose_from), 143, 288))
    for i in range(len(timesteps_to_choose_from)):
        image = one_obj.SSTA[timesteps_to_choose_from[i], :, :].values
        image = np.nan_to_num(image)

        # Calculate spatial autocorrelation
        image_product = np.fft.fft2(image) * np.fft.fft2(image).conj()
        sa_image = np.fft.fftshift(np.fft.ifft2(image_product))
        sa_image_array[i, :, :] = np.real(sa_image)
    return sa_image_array

def convert_from_timeres_to_months(time_step):
    """
    Converts the timestep to a month in the year
    Parameters
    ------------
    timestep : int 
    Returns
    ------------
    month
    """
    bymonth = np.resize(np.arange(1,13),12*166)[1:-11]
    month = bymonth[time_step]
    return month

def _get_labels(binary_images):
    blobs_labels = label_np(binary_images, background=0)
    return blobs_labels

def _get_centroids(sub_labels):
    props = regionprops(sub_labels.astype('int'))
    centroids = [(float(sub_labels.lat[round(p.centroid[0])].values),
                  float(sub_labels.lon[round(p.centroid[1])].values)) for p in props]
    for i in range(0,len(centroids)):
        if centroids[i][1] >= 359.75:
            centroids[i] = (centroids[i][0], list(centroids[i])[1] - 359.75)
    return centroids

def _get_center_of_mass(intensity_image):
    img = intensity_image.fillna(0)
    com = ndimage.center_of_mass(img.data)
    w_centroids = [(float(img.lat[round(com[0])].values),
                  float(img.lon[round(com[1])].values))]
    for i in range(0,len(w_centroids)):
        if w_centroids[i][1] >= 359.75:
            w_centroids[i] = (w_centroids[i][0], list(w_centroids[i])[1] - 359.75)
    return w_centroids

def plot_displacement(coordinate_list, intensity_array):
    summed_over_all_timesteps = intensity_array.sum(axis=0)
    summed_over_all_timesteps = xr.where(summed_over_all_timesteps == 0., 
                                         np.nan, 
                                         summed_over_all_timesteps)
    
    y_val_cent = list(zip(*coordinate_list))[0]; x_val_cent = list(zip(*coordinate_list))[1]
    x = x_val_cent; dx = [j-i for i, j in zip(x[:-1], x[1:])]
    y = y_val_cent; dy = [j-i for i, j in zip(y[:-1], y[1:])]

    plt.rcParams["figure.figsize"] = (12,6)
    summed_over_all_timesteps.plot.imshow(cmap='Greys',vmin=0,vmax=5)
    plt.quiver(x[:len(x)-1],y[:len(x)-1], dx,dy, width=0.005, color='Orange')
    for i in range(len(x)):
        plt.text(x[i],y[i],i, fontsize=15,c='Red')
        plt.scatter(x[i],y[i], c='Red')
    plt.scatter(x_val_cent[0], y_val_cent[0],c='Red'); 
    plt.scatter(x_val_cent[-1], y_val_cent[-1],c='Red')
    
def centroids_per_timestep(forOneMHW_onlylabels_timesteps, timestep):
    timestep_of_interest = forOneMHW_onlylabels_timesteps[timestep,:,:]
    get_sub_lbs = timestep_of_interest
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
            sub_labels_new = _get_labels(append_east_binarized)
            sub_labels_new = xr.DataArray(sub_labels_new, dims=append_east_binarized.dims, coords=append_east_binarized.coords)
            if len(np.unique(sub_labels_new)) <= 2:
                sub_labels_new = sub_labels_new.where(append_east_binarized>0, drop=False, other=np.nan)
                centroid_list.append(_get_centroids(sub_labels_new)) 
    flat_centroid_list = list(set([item for sublist in centroid_list for item in sublist])) # try saving as an xarray or data_array
    return flat_centroid_list

def displacement(forOneMHW_onlylabels_timesteps, forOneMHW_onlySSTA_timesteps):
    """
    Returns
    ----------
    - 'centroid_coords': a list of tuples with the centroid coordinates for each timestep.
    - 'com_coords': a list of tuples with the center of mass coordinates for each timestep.
    - 'centroid_displacements': a list of distances between centroid coordinates for consecutive timesteps.
    - 'com_displacements': a list of distances between center of mass coordinates for consecutive timesteps.
    - 'centroid_xrcoords': a list of xarray coordinates for the centroid for each timestep.
    - 'com_xrcoords': a list of xarray coordinates for the center of mass for each timestep.
    """
    centroid_list = []; centroid_xrcoords_ls = []
    center_of_mass_list = []; com_xrcoords_ls = []
    for i in range(forOneMHW_onlylabels_timesteps.shape[0]):
        forOneMHW_onlylabels_timesteps = xr.where(forOneMHW_onlylabels_timesteps > 0, 1, np.nan)
        forOneMHW_onlySSTA_timesteps = xr.where(forOneMHW_onlySSTA_timesteps > 0, 1, np.nan)
        img_cent_xr_coords = _get_center_of_mass(forOneMHW_onlylabels_timesteps[i,:,:])
        centroid_xrcoords_ls.append(img_cent_xr_coords[0])
        img_SSTA_xr_coords = _get_center_of_mass(forOneMHW_onlySSTA_timesteps[i,:,:])
        com_xrcoords_ls.append(img_SSTA_xr_coords[0])
        img_cent = forOneMHW_onlylabels_timesteps[i,:,:].fillna(0)
        img_SSTA = forOneMHW_onlySSTA_timesteps[i,:,:].fillna(0)
        centroid_list.append(ndimage.center_of_mass(img_cent.data))
        center_of_mass_list.append(ndimage.center_of_mass(img_SSTA.data))

    y_val_cent = list(zip(*centroid_list))[0]; x_val_cent = list(zip(*centroid_list))[1]
    y_val_com = list(zip(*center_of_mass_list))[0]; x_val_com = list(zip(*center_of_mass_list))[1]

    convert_long_range = interp1d([0,288],[-180,180])
    convert_lat_range = interp1d([0,143],[-65,70])

    coords_cent = list(zip(convert_lat_range(y_val_cent), convert_long_range(x_val_cent)))
    coords_com = list(zip(convert_lat_range(y_val_com), convert_long_range(x_val_com)))

    distance_cent_ls = []; distance_com_ls = []
    for i in range(len(coords_cent)-1):
        distance_cent = haversine(coords_cent[i], coords_cent[i+1],Unit.KILOMETERS)
        distance_cent_ls.append(distance_cent)
        distance_com = haversine(coords_com[i], coords_com[i+1],Unit.KILOMETERS)
        distance_com_ls.append(distance_com)
    return centroid_list, center_of_mass_list, distance_cent_ls, distance_com_ls, centroid_xrcoords_ls, com_xrcoords_ls

# this is the one
def com_per_timestep(forOneMHW_onlylabels_timesteps, forOneMHW_onlySSTA_timesteps, timestep):
    timestep_of_interest = forOneMHW_onlylabels_timesteps[timestep,:,:]
    SSTA_in_timestep = forOneMHW_onlySSTA_timesteps[timestep,:,:]
    
    sub_labels = _get_labels(timestep_of_interest) # use skimage to get sub_labels
    sub_labels = xr.DataArray(sub_labels, dims=timestep_of_interest.dims, coords=timestep_of_interest.coords)
    sub_labels = sub_labels.where(timestep_of_interest>0, drop=False, other=np.nan)
    
    edge_right_sub_labels = np.unique(np.unique(sub_labels[:,-1:])[~np.isnan(np.unique(sub_labels[:,-1:]))])
    edge_left_sub_labels = np.unique(np.unique(sub_labels[:,:1])[~np.isnan(np.unique(sub_labels[:,:1]))])
    edge_labels = np.unique(np.concatenate((edge_right_sub_labels, edge_left_sub_labels)))
    nonedgecases = np.setdiff1d(np.unique(sub_labels), edge_labels)
    nonedgecases = np.unique(nonedgecases[~np.isnan(nonedgecases)])

    com_list = []
    for i in nonedgecases:
        sub_labels_nonedgecases = xr.where(sub_labels==i, SSTA_in_timestep, np.nan)
        #sub_labels_nonedgecases.plot(); plt.show()
        sub_labels_nonedgecases_labels = sub_labels.where(sub_labels==i, drop=False, other=np.nan)
        com_list.append(_get_center_of_mass(sub_labels_nonedgecases)[0])
    for i in edge_left_sub_labels:
        sub_labels_left = sub_labels.where(sub_labels==i, drop=True)
        lon_edge = sub_labels_left[:,-1:].lon.item()
        if lon_edge < 358.75:
            SSTA_left = SSTA_in_timestep.where((SSTA_in_timestep.lon <= lon_edge), drop=True)
            #SSTA_left.plot(); plt.show()
            sub_labels_left.coords['lon'] = (sub_labels_left.coords['lon'] + 360) 
            SSTA_left.coords['lon'] = (SSTA_left.coords['lon'] + 360) 
            #SSTA_left.plot(); plt.show()
            for j in edge_right_sub_labels:
                sub_labels_right = sub_labels.where(sub_labels==j, drop=False, other=np.nan)
                sub_SSTAs_right = SSTA_in_timestep.where(sub_labels==j, drop=False, other=np.nan)
                east = sub_labels_right.where(sub_labels_right.lon > lon_edge, drop=True)
                east_SSTA = sub_SSTAs_right.where(sub_SSTAs_right.lon > lon_edge, drop=True)
                append_east = xr.concat([east.where(east.lon >= lon_edge, drop=True), sub_labels_left], dim="lon")
                #append_east.plot(); plt.show()
                append_east_SSTA = xr.concat([east_SSTA.where(east_SSTA.lon >= lon_edge, drop=True), SSTA_left], dim="lon")
                #append_east_SSTA.plot(); plt.show()
                append_east_binarized = xr.where(append_east > 0, 1, np.nan)
                sub_labels_new = _get_labels(append_east_binarized)
                sub_labels_new = xr.DataArray(sub_labels_new, dims=append_east_binarized.dims, coords=append_east_binarized.coords)
                #sub_labels_new.plot(); plt.show()
                if len(np.unique(sub_labels_new)) <= 2:
                    sub_labels_new = sub_labels_new.where(append_east_binarized>0, drop=False, other=np.nan)
                    #sub_labels_new.plot(); plt.show()
                    sub_labels_new_SSTA = append_east_SSTA.where(append_east_binarized>0, drop=False, other=np.nan)
                    #sub_labels_new.plot(); plt.show()
                    #sub_labels_new_SSTA.plot(); plt.show()
                    com_list.append(_get_center_of_mass(sub_labels_new_SSTA)[0])
    return com_list