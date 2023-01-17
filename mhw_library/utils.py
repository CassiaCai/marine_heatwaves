import xarray as xr

def create_events_file(detrended, blobs):
    '''
    Merges SST anomalies with the MHW identifier labels.
    Parameters
    ----------
    detrended : xarray.DataArray (time:1980, lat: 192, lon: 288)
        Consists of SST anomalies.
    blobs : xarray.DataArray 'labels' (time: 1980, lat: 192, lon: 288)
        Consists of MHW identifier labels.
    '''
    detrended.name = 'SSTA'
    return xr.merge([detrended, blobs])