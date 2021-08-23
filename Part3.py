import random
import numpy as np
import matplotlib.pyplot as plt

#TODO: Get the number of peaks you want to plot
number_of_peaks = int(input("Enter the Number of Peaks You want to Plot: "))
# list_of_random number ranges
random_range_list = []

#TODO: Generate the range of random lists and store them in the random number lists
for i in range(number_of_peaks):
    random_range_list.append(np.random.uniform(i, i+1, 100))
    for j in range(100):
        random_range_list[i][j] = round(random_range_list[i][j], 2)
print(random_range_list)

#TODO: After Generating the list of ranges plot the histogram of these ranges in matplotlib in same graph
bins = np.linspace(-1*number_of_peaks, number_of_peaks, 200)
for i in random_range_list:
    plt.hist(i, bins, alpha=0.5, label="Plot "+str(i))
plt.show()