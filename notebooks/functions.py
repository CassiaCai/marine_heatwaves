import numpy as np
from skimage import img_as_float
from skimage.morphology import convex_hull_image
from skimage.measure import label as label_np, regionprops
from scipy import ndimage

def apply_ocetrac_to_CESM2LE(member_id_val=0, threshold_val=0.9, radius_val=3, min_size_quartile_val=0.75, start_val=0, end_val=1980):
    """
    Apply ocetrac (https://ocetrac.readthedocs.io/en/latest/) to
    CESM2 LE, which has 100 ensemble members. The two main Ocetrac 
    parameters are radius_val and min_size_quartile_val which can be
    tuned according to data resolution and distribution.
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
    subset = coll_orig.search(component='atm',variable='SST',frequency='month_1',experiment='historical',member_id= str(member_id_list[1]))
    dsets = subset.to_dataset_dict(zarr_kwargs={"consolidated": True}, storage_options={"anon": True})
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
    model_da = xr.DataArray(model.T, dims=['time','coeff'], coords={'time':SST.time.values[start_val:end_val], 'coeff':np.arange(1,7,1)}) 
    pmodel_da = xr.DataArray(pmodel.T, dims=['coeff','time'], coords={'coeff':np.arange(1,7,1), 'time':SST.time.values[start_val:end_val]})
    # resulting coefficients of the model
    sst_mod = xr.DataArray(pmodel_da.dot(SST), dims=['coeff','lat','lon'], coords={'coeff':np.arange(1,7,1), 'lat':SST.lat.values, 'lon':SST.lon.values})
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
    Tracker = ocetrac.Tracker(binary_out_afterlandmask[:,:,:], newmask, radius=radius_val, min_size_quartile=min_size_quartile_val, timedim = 'time', xdim = 'lon', ydim='lat', positive=True)
    blobs = Tracker.track()
    blobs.attrs
    mo = Tracker._morphological_operations()
    return detrended, blobs

def create_events_file(detrended, blobs):
    """
    Merges SST anomalies with the MHW identifier labels.
    Parameters
    ------------
    detrended : xarray.DataArray (time:1980, lat: 192, lon: 288)
        Consists of SST anomalies.
    blobs : xarray.DataArray 'labels' (time: 1980, lat: 192, lon: 288)
        Consists of MHW identifier labels.
    Returns
    ------------
    xarray.DataArray with data variables SSTA and labels
    """
    detrended.name = 'SSTA'
    return xr.merge([detrended, blobs])

def number_of_mhws(event_file: xarray.Dataset) -> int:
    """
    Find the total number of objects in an event file
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    Returns
    ------------
    Integer: total number of objects in an event file
    """
    return len(np.unique(event_file.labels)) - 1

def calc_duration(event_file: , mhw_id):
    """
    Calculates the number of timesteps the object identifier
    appears in the event_file
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
    Integer: total number of timesteps where the object identifier appears
    """
    return len(event_file.where(event_file.labels==mhw_id, drop=True).time)

def calc_cumulativeintensity(event_file, mhw_id):
    """
    Calculates the (1) cumulative intensity of the object as
    a sum of all its grid points over all timesteps and (2) 
    cumulative intensity of the object at each timestep
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. cumulative_intensity 
        2. cumulative_intensity_monthly  
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    cumulative_intensity = for_one_mhw.SSTA.sum()
    cumulative_intensity_monthly = for_one_mhw.SSTA.sum(axis=(1,2))
    return cumulative_intensity, cumulative_intensity_monthly #cumulative_intensity_monthly.values

def calc_meanintensity(event_file, mhw_id) -> Tuple:
    """
    Calculates the (1) mean intensity of the object as a 
    mean of all its grid points over all timesteps and (2)
    mean intensity of the object at each timestep
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. mean_intensity 
        2. mean_intensity_monthly 
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    mean_intensity = for_one_mhw.mean() #np.nanmean(for_one_mhw.SSTA)
    mean_intensity_monthly = for_one_mhw.SSTA.mean(axis=(1,2))
    return mean_intensity.SSTA, mean_intensity_monthly # mean_intensity_monthly.values

def calc_maximumintensity(event_file, mhw_id):
    """
    Calculates the (1) maximum intensity of the object as a
    maximum of all its grid points over all timesteps and (2)
    maximum intensity of the object at each timestep
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. max_intensity
        2. max_intensity_monthly
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    max_intensity = for_one_mhw.max()
    max_intensity_monthly = for_one_mhw.SSTA.max(axis=(1,2))
    return max_intensity.SSTA, max_intensity_monthly

def calc_stdintensity(event_file, mhw_id) -> Tuple:
    """
    Calculates the (1) standard deviation (stdev) of the intensity of the
    object as a stdev of all its grid points over all timesteps and (2)
    as a stdev of the object at each timestep
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. std_intensity
        2. std_intensity_monthly
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    std_intensity = for_one_mhw.std()
    std_intensity_monthly = for_one_mhw.SSTA.std(axis=(1,2))
    return std_intensity.SSTA, std_intensity_monthly

def calc_spatialextent(event_file, mhw_id):
    """
    Describes the spatial extent of an object
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. coords_full : coordinates where the object has an imprint
        2. spatial_extents 
        3. max_spatial_extent
        4. max_spatial_extent_time
        5. mean_spatial_extent
        6. cumulative_spatial_extent
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    spatial_extents = []
    coords_full = []
    for i in range(len(for_one_mhw.time)):
        for_onetimestep_stacked = for_one_mhw.labels[i,:,:].stack(zipcoords=['lat','lon'])
        intermed = for_onetimestep_stacked[for_onetimestep_stacked.notnull()].zipcoords.values
        lats = [x[0] for x in intermed]; lons = [x[1] for x in intermed]
        coords = list(zip(lats, lons))
        coords_full.append(coords)
        y,x=zip(*coords)
        dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]; dlat = (110.574 *1) * np.ones(len(dlon))
        area = np.sum(dlon*dlat)
        spatial_extents.append(area)
    max_spatial_extent = np.max(spatial_extents)
    max_spatial_extent_time = np.argmax(spatial_extents)
    mean_spatial_extent = np.mean(spatial_extents)
    cumulative_spatial_extent = np.sum(spatial_extents)
    return coords_full, spatial_extents, max_spatial_extent, max_spatial_extent_time, mean_spatial_extent, cumulative_spatial_extent

def initialization(event_file, mhw_id):
    """
    Gets the initialization state information
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. Timestep when the object first appears
        2. Image of the object
        3. Month when the object initializes
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(for_one_mhw.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    bymonth = np.resize(np.arange(1,13),12*166)[1:-11]
    month = bymonth[first_timestep]
    return first_timestep, for_one_mhw.SSTA[first_timestep,:,:], month

from skimage.measure import find_contours
from haversine import haversine, Unit
from scipy.interpolate import interp1d

def calc_perimeter(event_file, mhw_id):
    """
    Calculates the perimeter of the object at each timestep
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. perimeter_ls
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=False)
    first_timestep, first_array, month = initialization(event_file, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)

    convert_long_range = interp1d([0,360],[-180,180])
    perimeter_ls = []
    for i in timesteps_to_choose_from:
        bw = for_one_mhw.labels[i,:,:].values > 0
        contours = find_contours(bw)
        distance_ls = []
        for contour_num in range(len(contours)):
            latitudes = for_one_mhw.lat.values[contours[contour_num][:,0].astype(int)]
            longitudes = for_one_mhw.lon.values[contours[contour_num][:,1].astype(int)]    
            coords = list(zip(latitudes, convert_long_range(longitudes)))

            for i in range(len(coords)-1):
                distance = haversine(coords[i], coords[i+1],Unit.KILOMETERS)
                distance_ls.append(distance)
            distance_ls.append(haversine(coords[len(coords)-1], coords[0],Unit.KILOMETERS))
        perimeter = np.sum(distance_ls)
        perimeter_ls.append(perimeter)
    return perimeter_ls  

def calc_percperimetervsarea(spatial_extents, perimeters):
    """
    Calculates the perimeter versus area percentage of an object
    at each timestep 
    Parameters
    ------------
    spatial_extents : list of areas
    perimeters : list of perimeters
    Returns
    ------------
    np.ndarray of percentages of perimeter versus spatial extent
    Notes
    ------------
    Gives an idea of how deformed an object is
    """
    return (np.asarray(perimeters)/np.asarray(spatial_extents))*100

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
    month = bymonth[first_timestep]
    return month

def calc_compltodeform(coords_full, spatial_extents):
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
    """
    perc_sharedarea_ls = []
    for i in range(len(coords_full)-1):
        a_set = set(coords_full[i])
        b_set = set(coords_full[i+1])
        if a_set & b_set:
            coords = a_set & b_set
            y,x=zip(*coords)
            dlon = [np.cos(y[c]*np.pi/180)*(111.320*1) for c in np.arange(0, len(coords))]; dlat = (110.574 *1) * np.ones(len(dlon))
            sharedarea = np.sum(dlon*dlat)
            perc_sharedarea_ls.append((sharedarea/ (spatial_extents[i] + spatial_extents[i+1]))*100)
        else:
            sharedareaarea = 0
            perc_sharedarea_ls.append((sharedarea/ (spatial_extents[i] + spatial_extents[i+1]))*100)
    return perc_sharedarea_ls

def calc_deform(perc_sharedarea_ls):
    """
    Calculates the deformation, which is the fraction of non-overlapped 
    domain occupied by a MHW event at 2 different times
    Parameters
    ------------
    perc_sharedarea_ls : list
    Returns
    ------------
    np.ndarray : fraction of non-overlapped domain occupied by a MHW event at
    each timestep
    """
    return np.asarray(100 - np.asarray(perc_sharedarea_ls))

def calc_whenlargesmall(spatial_extents):
    """
    Finds the timestep with (1) the largest spatial extent and (2)
    the smallest spatial extent
    Parameters
    ------------
    spatial_extents : object areas at each timestep
    Returns
    ------------
        1. when_large
        2. when_small 
    """
    when_large = (np.argmax(spatial_extents) / len(spatial_extents))*100
    when_small = (np.argmin(spatial_extents) / len(spatial_extents))*100
    return when_large, when_small

def calc_cross_correlation_spat(event_file, mhw_id):
    """
    Calculates the spatial cross correlation of an object
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. cc_image_array
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=False)
    first_timestep, first_array, month = initialization(event_file, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    cc_image_array = np.zeros((len(timesteps_to_choose_from), 192,288))    
    for i in range(len(timesteps_to_choose_from[:-1])):
        image = for_one_mhw.SSTA[timesteps_to_choose_from[i],:,:].values
        image = np.nan_to_num(image)
        offset_image = for_one_mhw.SSTA[timesteps_to_choose_from[i+1],:,:].values
        offset_image = np.nan_to_num(offset_image)
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        cc_image_array[i,:,:] = np.real(cc_image)
    return cc_image_array

def calc_perc_imoverchull(event_file, mhw_id):
    """
    Calculates the ratio of the convex hull area and the object area
    as a percentage
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. perc_imoverchull_ls
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=True)
    perc_imoverchull_ls = []
    for i in range(len(for_one_mhw.time)):
        image = for_one_mhw.labels[i].values
        image = [image == mhw_id][0]
        chull = convex_hull_image(image)
        chull_asflt = img_as_float(chull.copy())
        image_asflt = img_as_float(image.copy())
        perc_imoverchull = np.sum(image_asflt)/np.sum(chull_asflt)*100
        perc_imoverchull_ls.append(perc_imoverchull)
    return perc_imoverchull_ls

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
    props = regionprops(sub_labels.astype('int'))
    centroids = [(float(sub_labels.lat[round(p.centroid[0])].values),
                  float(sub_labels.lon[round(p.centroid[1])].values)) for p in props]
    for i in range(0,len(centroids)):
        if centroids[i][1] >= 359.75:
            centroids[i] = (centroids[i][0], list(centroids[i])[1] - 359.75)
    return centroids

def forOneMHW_labels_only(event_file, mhw_id):
    """
    Gets the labels for one object and all its timesteps
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. forOneMHW_only_timesteps
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(for_one_mhw.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = calc_duration(event_file, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = for_one_mhw.labels[timesteps_to_choose_from,:,:]
    return forOneMHW_only_timesteps

def forOneMHW_SSTA_only(event_file, mhw_id):
    """
    Gets the sea surface temperature anomalies for one object 
    and all its timesteps
    Parameters
    ------------
    event_file : xarray.Dataset with data variables SSTA and labels
    mhw_id : specific object identifier
    Returns
    ------------
        1. forOneMHW_only_timesteps
    """
    for_one_mhw = event_file.where(event_file.labels==mhw_id, drop=False)
    mhw_when = np.argwhere(for_one_mhw.labels.max(axis=(1,2)).values > 0.)
    first_timestep = mhw_when[0][0]
    duration = calc_duration(event_file, mhw_id)
    timesteps_to_choose_from = np.arange(first_timestep, first_timestep+duration)
    forOneMHW_only_timesteps = for_one_mhw.SSTA[timesteps_to_choose_from,:,:]
    return forOneMHW_only_timesteps

def centroids_per_timestep(forOneMHW_onlylabels_timesteps, timestep):
    """
    Finds the locations of the centroids of an object at each timestep
    Parameters
    ------------
    forOneMHW_onlylabels_timesteps : xarray.DataArray
    timestep : int
    Returns
    ------------
        1. flat_centroid_list
    Notes
    ------------
    There can be more than 1 centroid per object per timestep.
    """
    # Step 1. We start with one timestep and get all the sublabels
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
            sub_labels = _get_labels(append_east_binarized)
            sub_labels = xr.DataArray(sub_labels, dims=append_east_binarized.dims, coords=append_east_binarized.coords)
            sub_labels = sub_labels.where(append_east_binarized>0, drop=False, other=np.nan)
            centroid_list.append(_get_centroids(sub_labels))
    flat_centroid_list = list(set([item for sublist in centroid_list for item in sublist]))
    return flat_centroid_list

def _get_center_of_mass(intensity_image):
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

def com_per_timestep(forOneMHW_onlylabels_timesteps, forOneMHW_onlySSTA_timesteps, timestep):
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
    """
    timestep_of_interest = forOneMHW_onlylabels_timesteps[timestep,:,:] # labels in one given timestep
    SSTA_in_timestep = forOneMHW_onlySSTA_timesteps[timestep,:,:] # SSTA in one given timestep

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

def displacement(forOneMHW_onlylabels_timesteps, forOneMHW_onlySSTA_timesteps):
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

    convert_long_range = interp1d([0,360],[-180,180]); convert_lat_range = interp1d([0,180],[-90,90])

    coords_cent = list(zip(convert_lat_range(x_val_cent), convert_long_range(y_val_cent)))
    coords_com = list(zip(convert_lat_range(x_val_com), convert_long_range(y_val_com)))

    distance_cent_ls = []; distance_com_ls = []
    for i in range(len(coords_cent)-1):
        distance_cent = haversine(coords_cent[i], coords_cent[i+1],Unit.KILOMETERS)
        distance_cent_ls.append(distance_cent)
        distance_com = haversine(coords_com[i], coords_com[i+1],Unit.KILOMETERS)
        distance_com_ls.append(distance_com)
    return centroid_list, center_of_mass_list, distance_cent_ls, distance_com_ls, centroid_xrcoords_ls, com_xrcoords_ls