import numpy as np
import pandas as pd
from math import isnan
from matplotlib import pyplot as plt

# calibration = np.array()
file = pd.read_csv("values.csv", delimiter=" ")
file = file.iloc[:, 1:]
file = file.values
length = len(file)
for row in range(length):
    if isnan(file[row][0]):
        threshold_values = row + 1
        detection_values = row + 3
        calibration_values = row - 1
        break
calibration_values_emg = []
calibration_values_imu = []
threshold_values_emg = file[threshold_values][0:2]
threshold_values_imu = file[threshold_values][3]
print(threshold_values_emg)
print(threshold_values_imu)
detection_values_emg = []
detection_values_imu = []
for row in range(calibration_values + 1):
    calibration_values_imu.append(file[row][0:2])
    calibration_values_emg.append(file[row][3])
for row in range(detection_values, length):
    detection_values_imu.append(file[row][0:2])
    detection_values_emg.append(file[row][3])
calibration_values_emg = np.array(calibration_values_emg)
calibration_values_imu = np.array(calibration_values_imu)
detection_values_emg = np.array(detection_values_emg)
detection_values_imu = np.array(detection_values_imu)
plt.subplot(2, 2, 1)
plt.plot(calibration_values_emg)
plt.plot(threshold_values_emg)
plt.title("Calibration Values EMG")
plt.subplot(2, 2, 2)
plt.plot(calibration_values_imu)
plt.title("Calibration Values IMU")
plt.subplot(2, 2, 3)
plt.plot(detection_values_emg)
plt.title("Detection Values EMG")
plt.subplot(2, 2, 4)
plt.plot(detection_values_imu)
plt.title("Detection Values IMU")
plt.show()
