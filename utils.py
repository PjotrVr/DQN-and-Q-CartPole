import matplotlib.pyplot as plt
import numpy as np


def plot_training(x_label, y_label, values, legend_labels, colors, average_freq, xlim, ylim, figsize=(12, 10), filename=None):
    assert len(values) == len(legend_labels) == len(colors)
    
    # Creating a figure
    fig = plt.figure(figsize=figsize)
    plt.xlabel(x_label, color="black")
    plt.ylabel(y_label, color="black")
    
    for value, label, color in zip(values, legend_labels, colors):
        value_len = len(value)
        running_average = np.empty(value_len)
        for timestep in range(value_len):
            running_average[timestep] = np.mean(value[max(0, timestep-average_freq):timestep+1])
        
        plt.plot([x_value for x_value in range(value_len)], running_average, color=color, label=label)
        
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.legend()

    # Save if needed
    if filename is not None:
        plt.savefig(filename)
        
    plt.show()
    
    