import numpy as np

def toy_example(num_timesteps: int, lat_len: int, lon_len: int, mu: float, sigma: float) -> np.ndarray:
    """ Create simple toy example of an event. No splitting.
    Args:
        num_timesteps (int): Event length.
        lat_len (int): Resolution.
        lon_len (int): Resolution.
        Mu (float): Mean.
        Sigma (float): Standard deviation.
    Returns:
        np.ndarray: Array of shape (lat_len, lon_len, num_timesteps).
    """
    toy_temp_array = np.zeros(shape=(lat_len, lon_len, num_timesteps))

    for i in range(0, num_timesteps):
        randint_1 = np.random.randint(10,np.round(lat_len/3)); randint_2 = np.random.randint(10,np.round(lon_len/3))
        temp_array = np.random.normal(mu, sigma, size=(randint_1, randint_2))
        
        pad_1 = np.random.randint(2,np.round(lat_len/3)); pad_2 = np.random.randint(2,np.round(lat_len/3))
        pad_3 = np.random.randint(2,np.round(lat_len/3)); pad_4 = np.random.randint(2,np.round(lat_len/3))
        temp_array=np.pad(temp_array, ((pad_1,pad_2),(pad_3,pad_4)), 'constant')
        
        x_pad = lat_len - temp_array.shape[0]; y_pad = lon_len - temp_array.shape[1]
        temp_array=np.pad(temp_array, ((x_pad,0),(0,y_pad)), mode='constant', constant_values=0)
        
        toy_temp_array[:,:,i] = temp_array
    return toy_temp_array

def toy_example_withsplit(num_timesteps: int, lat_len: int, lon_len: int, mu: float, sigma: float) -> np.ndarray:
    """ An event that splits into multiple spots and stays split up
    Args:
        num_timeres (int): Number of available timesteps of data.
        num_timesteps (int): Event length.
        lat_len (int): Resolution.
        lon_len (int): Resolution.
        Mu (float): Mean.
        Sigma (float): Standard deviation.
    Returns:
        np.ndarray: Array of shape (lat_len, lon_len, num_timesteps).
    """
    when_to_split = np.random.randint(1,num_timesteps-2)
    when_to_merge = np.random.randint(when_to_split+1,num_timesteps)
    toy_temp_array = np.zeros(shape=(lat_len, lon_len, num_timesteps))

    for i in list(range(0, when_to_split)) + list(range(when_to_merge, num_timesteps)):
        randint_1 = np.random.randint(10,np.round(lat_len/3)); randint_2 = np.random.randint(10,np.round(lon_len/3))
        temp_array = np.random.normal(mu, sigma, size=(randint_1, randint_2))
        pad_1 = np.random.randint(2,np.round(lat_len/3)); pad_2 = np.random.randint(2,np.round(lat_len/3))
        pad_3 = np.random.randint(2,np.round(lat_len/3)); pad_4 = np.random.randint(2,np.round(lat_len/3))
        temp_array=np.pad(temp_array, ((pad_1,pad_2),(pad_3,pad_4)), 'constant')
        x_pad = lat_len - temp_array.shape[0]; y_pad = lon_len - temp_array.shape[1]
        temp_array=np.pad(temp_array, ((x_pad,0),(0,y_pad)), mode='constant', constant_values=0)
        toy_temp_array[:,:,i] = temp_array

    for i in range(when_to_split, when_to_merge):
        randint_1 = np.random.randint(10,np.round(lat_len/3)); randint_2 = np.random.randint(10,np.round(lon_len/3))
        temp_array = np.random.normal(mu, sigma, size=(randint_1, randint_2))
        split = np.random.randint(0,int(temp_array.shape[0]/2))
        split_width = np.random.randint(int(temp_array.shape[0]/2),int(temp_array.shape[0]/2)+5 )
        temp_array[split:split_width,:] = 0.
        pad_1 = np.random.randint(2,np.round(lat_len/3)); pad_2 = np.random.randint(2,np.round(lat_len/3))
        pad_3 = np.random.randint(2,np.round(lat_len/3)); pad_4 = np.random.randint(2,np.round(lat_len/3))
        temp_array=np.pad(temp_array, ((pad_1,pad_2),(pad_3,pad_4)), 'constant')
        x_pad = lat_len - temp_array.shape[0]; y_pad = lon_len - temp_array.shape[1]
        temp_array=np.pad(temp_array, ((x_pad,0),(0,y_pad)), mode='constant', constant_values=0)
        toy_temp_array[:,:,i] = temp_array
    return toy_temp_array

def toy_example_into_full_time(num_timeres: int, toy_temp_array: np.ndarray) -> np.ndarray:
    """ Toy example placed somewhere within the full time array
    Args:
        num_timeres (int): Number of available timesteps of data.
    Returns:
        np.ndarray: Array of shape (lat_len, lon_len, num_timeres).
    """
    full_toy_temp_array = np.zeros(shape=(toy_temp_array.shape[0], toy_temp_array.shape[1], num_timeres))
    randinit_time = np.random.randint(0,num_timeres-toy_temp_array.shape[2])
    full_toy_temp_array[:,:,randinit_time:randinit_time+toy_temp_array.shape[2]] = toy_temp_array
    return full_toy_temp_array

def const_toy_example(toy_temp_array: np.ndarray, const_tval: float) -> np.ndarray:
    """ Use the above simple toy example and replace non-zero values with a constant.
    Args:
        toy_temp_array (np.ndarray): Temperature measurements.
        const_tval (float): Some constant.
    Returns: 
        np.ndarray: Array of shape (lat_len, lon_len, num_timesteps).
    """
    toy_temp_array[toy_temp_array != 0] = const_tval
    return toy_temp_array