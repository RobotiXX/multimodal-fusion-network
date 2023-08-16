import numpy as np
import matplotlib.pyplot as plt

# Define the mean and standard deviation of the Gaussian distribution


def get_gaussian_weights(mean, std_dev):
    mean = 2.5
    std_dev = 1.4
    domain = np.array([i for i in range(0,12)])

    range_values = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((domain - mean) / std_dev)**2)
    
    # Output the values at integer points
    # for i in range(range_values.shape[0]):    
    #     print("Value at x =", i, "is", np.floor(range_values[i]*1000)+100)
    
    round_weights = np.floor(range_values * 1000)
    reshapeed_weights  = np.expand_dims(round_weights, axis=0)

    return  reshapeed_weights
