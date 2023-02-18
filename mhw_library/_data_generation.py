import numpy as np

def toy_example(
    num_timesteps: int, 
    lat_len: int, 
    lon_len: int, 
    mu: float, 
    sigma: float) -> np.ndarray:
    """
    Create a simple toy example of a weather event, represented as an array of temperature values.

    Parameters
    ----------
    num_timesteps : int
        The length of the event (in timesteps).
    lat_len : int
        The number of latitudes (i.e., rows) in the output array.
    lon_len : int
        The number of longitudes (i.e., columns) in the output array.
    mu : float
        The mean of the temperature values.
    sigma : float
        The standard deviation of the temperature values.

    Returns
    -------
    np.ndarray
        An array of shape (lat_len, lon_len, num_timesteps) containing the temperature values at each point in space and time.
    """

    # Input validation
    if not isinstance(num_timesteps, int) or num_timesteps < 1 \
            or not isinstance(lat_len, int) or lat_len < 1 \
            or not isinstance(lon_len, int) or lon_len < 1 \
            or not isinstance(mu, (int, float)) \
            or not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("Invalid input")

    # Create array of temperature values
    toy_temp_array = np.zeros(shape=(lat_len, lon_len, num_timesteps))

    for t in range(num_timesteps):
        # Generate random temperature values and padding
        rand_lat_len = np.random.randint(10, lat_len // 3)
        rand_lon_len = np.random.randint(10, lon_len // 3)
        pad_top, pad_left = np.random.randint(2, lat_len // 3, size=2)
        pad_bottom = lat_len - rand_lat_len - pad_top
        pad_right = lon_len - rand_lon_len - pad_left
        temp_array = np.pad(
            np.random.normal(mu, sigma, size=(rand_lat_len, rand_lon_len)),
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            'constant')

        # Add the temperature array to the output array
        toy_temp_array[..., t] = temp_array

    return toy_temp_array

def toy_example_withsplit(
    num_timesteps: int, 
    lat_len: int, 
    lon_len: int, 
    mu: float, 
    sigma: float,
    pad_min: int = 2,
    pad_max: int = 10,
    rand_min: int = 10,
    rand_max: int = 33,
    split_min: int = 5,
    split_max: int = 10
) -> np.ndarray:
    """
    Generate a toy example of an event that splits into multiple spots and stays split up.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps in the output array.
    lat_len : int
        Number of latitude points in the output array.
    lon_len : int
        Number of longitude points in the output array.
    mu : float
        Mean of the normal distribution used to generate the temperature values.
    sigma : float
        Standard deviation of the normal distribution used to generate the temperature values.

    Returns
    -------
    np.ndarray
        Array of shape (lat_len, lon_len, num_timesteps) containing the temperature values.
    """
    rng = np.random.default_rng()
    when_to_split = rng.integers(1, num_timesteps-2)
    when_to_merge = rng.integers(when_to_split+1, num_timesteps)
    toy_temp_array = np.zeros(shape=(lat_len, lon_len, num_timesteps))

    def generate_temp_array():
        randint_1 = rng.integers(rand_min, rand_max)
        randint_2 = rng.integers(rand_min, rand_max)
        temp_array = rng.normal(mu, sigma, size=(randint_1, randint_2))
        return temp_array

    for i in range(num_timesteps):
        if i < when_to_split or i >= when_to_merge:
            temp_array = generate_temp_array()
        else:
            temp_array = generate_temp_array()
            split = rng.integers(0, int(temp_array.shape[0]/2))
            split_width = rng.integers(int(temp_array.shape[0]/2), int(temp_array.shape[0]/2)+split_max)
            temp_array[split:split+split_width,:] = 0.

        pad_1 = rng.integers(pad_min, pad_max)
        pad_2 = rng.integers(pad_min, pad_max)
        pad_3 = rng.integers(pad_min, pad_max)
        pad_4 = rng.integers(pad_min, pad_max)
        temp_array = np.pad(temp_array, ((pad_1, pad_2), (pad_3, pad_4)), 'constant')

        x_pad = lat_len - temp_array.shape[0]
        y_pad = lon_len - temp_array.shape[1]
        temp_array = np.pad(temp_array, ((x_pad, 0), (0, y_pad)), mode='constant', constant_values=0)
        toy_temp_array[:,:,i] = temp_array

    return toy_temp_array

def insert_toy_example_into_full_time_array(num_timeres: int, toy_temp_array: np.ndarray) -> np.ndarray:
    """
    Inserts a toy example array into a full time array.

    Parameters
    ----------
    num_timeres : int
        The number of available timesteps of data in the full array.
    toy_temp_array : np.ndarray
        The toy example array to insert into the full time array. It should be of shape
        (lat_len, lon_len, num_timesteps) where num_timesteps is the number of timesteps in
        the toy example array and must be less than or equal to num_timeres.

    Returns
    -------
    np.ndarray
        An array of shape (lat_len, lon_len, num_timeres) where the toy example array is
        inserted at a randomly selected location in the time dimension. If the toy example
        array has fewer timesteps than the full time array, it is padded with zeros at the end.
    """
    full_toy_temp_array = np.zeros(shape=(toy_temp_array.shape[0], toy_temp_array.shape[1], num_timeres))
    randinit_time = np.random.randint(0,num_timeres-toy_temp_array.shape[2])
    full_toy_temp_array[:,:,randinit_time:randinit_time+toy_temp_array.shape[2]] = toy_temp_array
    return full_toy_temp_array

def const_toy_example(toy_temp_array: np.ndarray, const_tval: float) -> np.ndarray:
    """
    Replace all non-zero values in a toy example array with a constant value.

    Parameters
    ----------
    toy_temp_array : np.ndarray
        The toy example array to modify. It should be of shape (lat_len, lon_len, num_timesteps)
        and represent temperature measurements.
    const_tval : float
        The constant value to replace non-zero entries in the toy example array.

    Returns
    -------
    np.ndarray
        An array of shape (lat_len, lon_len, num_timesteps) where all non-zero values in the input
        array are replaced with the constant value.
    """
    toy_temp_array[toy_temp_array != 0] = const_tval
    return toy_temp_array