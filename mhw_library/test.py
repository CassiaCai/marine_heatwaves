import numpy as np
import xarray as xr
import pytest
import measures
import utils

def test_calc_duration():
    # Create test data
    ssta = xr.DataArray(np.random.rand(10, 10), dims=('time', 'space'))
    labels = xr.DataArray(np.random.randint(0, 2, size=(10, 10)), dims=('time', 'space'))
    merged_xarray = xr.Dataset({'SSTA': ssta, 'labels': labels})
    
    # Choose a specific object identifier to test
    mhw_id = 1
    
    # Set a few timestamps to have the mhw_id label
    labels[0, 0] = mhw_id
    labels[1, 2] = mhw_id
    labels[3, 5] = mhw_id
    
    # Call the function with the test data
    result = calc_duration(merged_xarray, mhw_id)
    
    # Assert that the result is correct
    assert result == 3
    
def test_calc_cumulative_intensity():
    # Create test data
    ssta = xr.DataArray(np.random.rand(10, 10), dims=('time', 'space'))
    labels = xr.DataArray(np.random.randint(0, 2, size=(10, 10)), dims=('time', 'space'))
    merged_xarray = xr.Dataset({'SSTA': ssta, 'labels': labels})
    
    # Choose a specific object identifier to test
    mhw_id = 1
    
    # Set some grid points to have the mhw_id label
    labels[0, 0] = mhw_id
    labels[1, 2] = mhw_id
    labels[2, 4] = mhw_id
    
    # Set some random SSTA values for the mhw_id label
    ssta[0, 0] = 1.5
    ssta[1, 2] = 2.5
    ssta[2, 4] = 3.5
    
    # Call the function with the test data
    result = calc_cumulative_intensity(merged_xarray, mhw_id)
    
    # Assert that the cumulative intensity is correct
    assert result[0] == 7.5
    
    # Assert that the cumulative intensity per timestamp is correct
    expected_cumulative_intensity_per_timestamp = np.array([1.5, 2.5, 3.5, 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(result[1].values, expected_cumulative_intensity_per_timestamp)

def test_calc_mean_intensity():
    # create sample xarray dataset
    data = np.random.rand(10, 5, 5)
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    ds = xr.Dataset(
        {"SSTA": (("time", "lat", "lon"), data), "labels": (("time"), labels)},
        coords={"time": pd.date_range("2000-01-01", periods=10), "lat": range(5), "lon": range(5)},
    )

    # calculate mean intensity for object 1
    mean_intensity, mean_intensity_per_timestamp = calc_mean_intensity(ds, 1)
    expected_mean_intensity = np.mean(ds.SSTA.where(ds.labels == 1, drop=True))
    expected_mean_intensity_per_timestamp = ds.SSTA.where(ds.labels == 1, drop=True).mean(dim=("lat", "lon"))

    # assert that calculated mean intensity is correct
    assert np.isclose(mean_intensity, expected_mean_intensity)

    # assert that calculated mean intensity per timestamp is correct
    assert np.allclose(mean_intensity_per_timestamp, expected_mean_intensity_per_timestamp)

def test_calc_std_intensity():
    # Generate mock data
    n_obj = 3
    n_time = 10
    n_lat = 20
    n_lon = 30
    data = np.random.rand(n_obj, n_time, n_lat, n_lon)
    labels = xr.DataArray(np.random.randint(1, n_obj + 1, size=(n_time, n_lat, n_lon)),
                          dims=('time', 'lat', 'lon'))
    ssta = xr.DataArray(data, dims=('object', 'time', 'lat', 'lon'))
    merged_xarray = xr.Dataset({'SSTA': ssta, 'labels': labels})

    # Calculate expected output
    mhw_id = 2  # Choose an arbitrary object id
    one_obj = merged_xarray.where(merged_xarray.labels==mhw_id, drop=True)
    std_intensity = one_obj.SSTA.std()
    std_intensity_per_timestamp = one_obj.SSTA.std(axis=(1,2))

    # Check function output
    calculated_std_intensity, calculated_std_intensity_per_timestamp = calc_std_intensity(merged_xarray, mhw_id)
    np.testing.assert_allclose(calculated_std_intensity, std_intensity)
    np.testing.assert_allclose(calculated_std_intensity_per_timestamp, std_intensity_per_timestamp)

