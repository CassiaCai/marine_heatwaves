import xarray as xr


def merged_by_objid(
    mask_field: xr.Dataset, 
    anomaly_field: xr.Dataset,
) -> xr.Dataset:
    """Merges the mask and anomaly fields for each obj
    
    Parameters:
        mask_field (xr.Dataset): The input xarray Dataset containing two values.
        anomaly_field (xr.Dataset): The input xarray Dataset containing a range of values.
    
    Returns:
        xr.Dataset with data variables mhw_number and SSTA
    """
    mask_field['mhw_number'] = mask_field['mhw_obj']
    anomaly_field['SSTA'] = anomaly_field['mhw_obj']
    
    mask_field = mask_field.drop(['mhw_obj'])
    anomaly_field = anomaly_field.drop(['mhw_obj'])
    
    merged_by_objid = xr.merge([mask_field, anomaly_field])
    return merged_by_objid

def measures_dict(
    unique_labels_list: list,
    ens_memb_ind: int,
) -> dict:
    
    data_dict = {
        'cumulative_intensity': [],
        'cumulative_intensity_monthly': [],
        'duration': [],
        'mean_intensity': [],
        'mean_intensity_monthly': [],
        'max_intensity': [],
        'max_intensity_monthly': [],
        'std_intensity': [],
        'std_intensity_monthly': [],
        'coords_full': [],
        'spatial_extents': [],
        'max_spatial_extent': [],
        'max_spatial_extent_time': [],
        'mean_spatial_extent': [],
        'cumulative_spatial_extent': [],
        'initialization_map': [],
        'perimeters': [],
        'complement_to_deform': [],
        'deformation': [],
        'mean_abs_sa': [],
        'std_sa': [],
        'sa_image_array_per_mhw': [],
        'imoverchull': [],
        'num_centroids': [],
        'flat_centroid_list': [],
        'center_of_mass_list': [],
        'centroid_one_list': [],
        'center_of_mass_one_list': [],
        'distance_cent_ls': [],
        'distance_com_ls': [],
        'centroid_xrcoords_ls': [],
        'com_xrcoords_ls': [],
        'first_time': []
    }
    
    for index in range(len(unique_labels_list[ens_memb_ind]) - 1):
        mhw_SSTA_mask_filename = f'/glade/derecho/scratch/cassiacai/mhw_ssta_objs/footprint_masks_SSTA_ens{ens_memb_ind}_mhw_{index}.nc'
        mhw_mask_filename = f'/glade/derecho/scratch/cassiacai/mhw_objs/footprint_masks_ens{ens_memb_ind}_mhw_{index}.nc'

        ens_mhw_SSTA_mask = xr.open_dataset(mhw_SSTA_mask_filename)
        ens_mhw_mask = xr.open_dataset(mhw_mask_filename)

        merged_file_by_mhwid = merged_by_objid(ens_mhw_mask, ens_mhw_SSTA_mask)

        # ---------
        cumulative_intensity = merged_file_by_mhwid.SSTA[:, :, :].sum()
        cumulative_intensity_monthly = merged_file_by_mhwid.SSTA[:, :, :].sum(dim=('lat','lon'))

        duration = len(merged_file_by_mhwid.SSTA.time)
        first_time = merged_file_by_mhwid.time[0].values

        mean_intensity = merged_file_by_mhwid.SSTA[:, :, :].mean()
        mean_intensity_monthly = merged_file_by_mhwid.SSTA[:, :, :].mean(dim=('lat','lon'))

        max_intensity = merged_file_by_mhwid.SSTA[:, :, :].max()
        max_intensity_monthly = merged_file_by_mhwid.SSTA[:, :, :].max(dim=('lat','lon'))

        std_intensity = merged_file_by_mhwid.SSTA[:, :, :].std()
        std_intensity_monthly = merged_file_by_mhwid.SSTA[:, :, :].std(dim=('lat','lon'))

        coords_full, spatial_extents, max_spatial_extent, max_spatial_extent_time, mean_spatial_extent, cumulative_spatial_extent = calc_spatialextent(merged_file_by_mhwid)
        initialization_map = merged_file_by_mhwid.SSTA.isel(time=0)

        perimeters_oneobj = calc_perimeter(merged_file_by_mhwid,np.arange(duration))
# only works for durations that are greater than 1 month
        if duration > 1:
            complement_to_deform = calc_complement_to_deformormation(coords_full, spatial_extents) 
            deformation = calc_deformation(complement_to_deform)

            imoverchull = calc_ratio_convexhullarea_vs_area(merged_file_by_mhwid, duration)
        else:
            complement_to_deform = 0
            deformation = 0
            imoverchull = 0

        # ----------------------- spatial autocorrelation
        sa_image_array_per_mhw = calc_spatial_cross_autocorrelation(merged_file_by_mhwid, np.arange(duration))

        std_image_array_per_mhw_intermed = []
        mean_image_array_per_mhw_intermed = []
        for i in range(duration):
            mean_abs_sa = abs(sa_image_array_per_mhw[i,:,:]).mean() # mean absolute value
            mean_image_array_per_mhw_intermed.append(mean_abs_sa)
            std_sa = sa_image_array_per_mhw[i,:,:].std() # standard deviation
            std_image_array_per_mhw_intermed.append(std_sa)

        forOneMHW_onlylabels_timesteps = merged_file_by_mhwid.mhw_number
        forOneMHW_onlySSTA_timesteps = merged_file_by_mhwid.SSTA

        centroid_one_list, center_of_mass_one_list, distance_cent_ls, distance_com_ls, centroid_xrcoords_ls, com_xrcoords_ls = displacement(forOneMHW_onlylabels_timesteps, 
                                                                                                                                            forOneMHW_onlySSTA_timesteps)

        num_centroids = []
        flat_centroid_list_intermed = []
        center_of_mass_list_intermed = []

        for timestep in range(duration):
            num_centroids.append(len(centroids_per_timestep(forOneMHW_onlylabels_timesteps, 
                                                                timestep)))
            flat_centroid_list_intermed.append(centroids_per_timestep(forOneMHW_onlylabels_timesteps, 
                                                                          timestep))  
            center_of_mass_list = com_per_timestep(forOneMHW_onlylabels_timesteps, 
                                                       forOneMHW_onlySSTA_timesteps,
                                                       timestep)        
            center_of_mass_list_intermed.append(center_of_mass_list)

        # Compute and store measures
        data_dict['cumulative_intensity'].append(cumulative_intensity)
        data_dict['cumulative_intensity_monthly'].append(cumulative_intensity_monthly)
        data_dict['duration'].append(duration)
        data_dict['mean_intensity'].append(mean_intensity)
        data_dict['mean_intensity_monthly'].append(mean_intensity_monthly)
        data_dict['max_intensity'].append(max_intensity)
        data_dict['max_intensity_monthly'].append(max_intensity_monthly)
        data_dict['std_intensity'].append(std_intensity)
        data_dict['std_intensity_monthly'].append(std_intensity_monthly)
        data_dict['coords_full'].append(coords_full)
        data_dict['spatial_extents'].append(spatial_extents)
        data_dict['max_spatial_extent'].append(max_spatial_extent)
        data_dict['max_spatial_extent_time'].append(max_spatial_extent_time)
        data_dict['mean_spatial_extent'].append(mean_spatial_extent)
        data_dict['cumulative_spatial_extent'].append(cumulative_spatial_extent)
        data_dict['initialization_map'].append(initialization_map)
        data_dict['perimeters'].append(perimeters_oneobj)
        data_dict['complement_to_deform'].append(complement_to_deform)
        data_dict['deformation'].append(deformation)
        data_dict['mean_abs_sa'].append(mean_image_array_per_mhw_intermed)
        data_dict['std_sa'].append(std_image_array_per_mhw_intermed)
        data_dict['sa_image_array_per_mhw'].append(sa_image_array_per_mhw)
        data_dict['imoverchull'].append(imoverchull)
        data_dict['num_centroids'].append(num_centroids)
        data_dict['flat_centroid_list'].append(flat_centroid_list_intermed)
        data_dict['center_of_mass_list'].append(center_of_mass_list_intermed)
        data_dict['centroid_one_list'].append(centroid_one_list)
        data_dict['center_of_mass_one_list'].append(center_of_mass_one_list)
        data_dict['distance_cent_ls'].append(distance_cent_ls)
        data_dict['distance_com_ls'].append(distance_com_ls)
        data_dict['centroid_xrcoords_ls'].append(centroid_xrcoords_ls)
        data_dict['com_xrcoords_ls'].append(com_xrcoords_ls)
        data_dict['first_time'].append(first_time)
    
    return data_dict
