
#   == IMPORTS ==
import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
import sys
import os.path


#   == SETTINGS ==
#   -- Searching --
search_delta = 2  # s
bio_zero_delta = 30  # s
baseline_delta = 30  # s

#   -- Filters --
low_filter_cutoff_freq = 1 / (2.0 * 3.14159)
high_filter_cutoff_freq = 2.0

#   -- Data Reading --
data_start_keyword = "F2"
data_col_names = ['No', 'hr', 'min', 'sec', 'F1', 'F2']
data_col_hr_index = 1
data_col_min_index = 2
data_col_sec_index = 3
data_col_y_index = 5

#   -- Rendering --
show_plots = True
show_interim = False


#   == FUNCTIONS ==
#   -- Command Line Arguments --
def get_args():
    if len(sys.argv) != 6:
        print("Invalid commanf line arguments. Use:")
        print("python3 final.py <patient_no> <data_file> <baseline_end> <cuff_release> <peak>")
        exit()
    patient_no = int(sys.argv[1])
    data_file = str(sys.argv[2])
    baseline_end_t = float(sys.argv[3])
    cuff_release_t = float(sys.argv[4])
    peak_t = float(sys.argv[5])
    return patient_no, data_file, baseline_end_t, cuff_release_t, peak_t


#   -- Parsing --
def find_data(filename, key_string):
    f = open(filename)
    line = f.readlines()
    f.close
    for i in range(len(line)):
        word = line[i].split()
        if len(word) > 0:
            if word[len(word) - 1] == key_string:
                return i


def get_data(filename, key_string, col_names):
    line_start = find_data(filename, key_string)
    if (line_start == None):
        print("Unable to locate", key_string, "within file", filename)
        exit()
    data = np.genfromtxt(filename, delimiter='\t', skip_header=(line_start + 1), names=col_names)
    return data


#   -- Data --
def create_time_arr(sec, min, hr):
    t = []
    for i in range(len(sec)):
        time = sec[i] + (60.0 * min[i]) + (3600.0 * hr[i])
        t.append(time)
    return t


#   -- Filters --
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


#   -- Turning Points --
def find_turning_points(x, y):
    tp_i = []
    tp_x = []
    tp_y = []
    for i in range(1, len(y) - 1):
        if (((y[i] > y[i - 1]) and (y[i] > y[i + 1])) or ((y[i] < y[i - 1]) and (y[i] < y[i + 1]))):
            tp_i.append(i)
            tp_x.append(x[i])
            tp_y.append(y[i])
    return tp_i, tp_x, tp_y


#   -- Searching --
def search_range(x, x0, delta):
    indices = []
    for i in range(len(x)):
        if ((x[i] >= (x0 - delta)) and (x[i] <= (x0 + delta))):
            indices.append(i)
    return indices


#   == MAIN ==
#   -- Command Line Arguments --
patient_no, data_file, user_baseline_end_t, user_cuff_release_t, user_peak_t = get_args()
print("Patient No.:", patient_no)
print("Data file:", data_file)
print("Baseline end time:", user_baseline_end_t, "s")
print("Est cuff release time:", user_cuff_release_t, "s")
print("Est peak time:", user_peak_t, "s")
print("---")


#   -- Parse Data --
data = get_data(data_file, data_start_keyword, data_col_names)
y = data[data_col_names[data_col_y_index]]
print("Data rows:", len(y))

t = create_time_arr(data[data_col_names[data_col_sec_index]],
                    data[data_col_names[data_col_min_index]],
                    data[data_col_names[data_col_hr_index]])
raw_data_freq = (len(t) - 1) / t[-1] - t[0]
print("Frequency:", raw_data_freq, "hz")
print("Start time:", t[0], "s")
print("End time:", t[-1], "s")
print("---")

if (show_plots and show_interim):
    plt.plot(t, y, 'r-', label='raw')
    plt.legend()
    plt.show()


#   -- Filter Data --
b, a = butter_lowpass(low_filter_cutoff_freq, raw_data_freq)
w, h = freqz(b, a, worN=8000)
y0 = butter_lowpass_filter(y, low_filter_cutoff_freq, raw_data_freq)

b, a = butter_lowpass(high_filter_cutoff_freq, raw_data_freq)
w, h = freqz(b, a, worN=8000)
y1 = butter_lowpass_filter(y, high_filter_cutoff_freq, raw_data_freq)

if (show_plots and show_interim):
    plt.plot(t, y, 'r-', label='raw')
    plt.plot(t, y1, 'm-', label='high')
    plt.plot(t, y0, 'b-', label='low')
    plt.legend()
    plt.show()


#   -- Turning Points --
tp0_i, tp0_t, tp0_y = find_turning_points(t, y0)
tp1_i, tp1_t, tp1_y = find_turning_points(t, y1)

if (show_plots and show_interim):
    plt.plot(t, y, 'r-', label='raw')
    plt.plot(t, y1, 'm-', label='high')
    plt.plot(tp1_t, tp1_y, 'mo', label='high tps')
    plt.plot(t, y0, 'b-', label='low')
    plt.plot(tp0_t, tp0_y, 'bo', label='low tps')
    plt.legend()
    plt.show()


#   -- Determine Cuff-Release --
cuff_indices = search_range(tp1_t, user_cuff_release_t, search_delta)
cuff_indices.insert(0, cuff_indices[0] - 1)
cuff_indices.append(cuff_indices[-1] + 1)
cuff_time = 0.0
cuff_delta = 0.0
cuff_index = -1
for i in range(cuff_indices[0], cuff_indices[-1] - 1):
    delta = tp1_y[i + 1] - tp1_y[i]
    if (delta > cuff_delta):
        cuff_delta = delta
        cuff_index = i
if (cuff_index == -1):
    print("ERROR! Unable to determine cuff-release time.")
    exit()
cuff_time = tp1_t[cuff_index]
print("Cuff time: ", cuff_time, "s")


#   -- Determine Peak Time --
peak_indices = search_range(tp0_t, user_peak_t, search_delta)
peak_indices.insert(0, peak_indices[0] - 1)
peak_indices.append(peak_indices[-1] + 1)
peak_time = 0.0
peak_height = 0.0
peak_index = -1
for i in range(peak_indices[0], peak_indices[-1]):
    delta = tp1_y[i + 1] - tp1_y[i]
    if (tp0_y[i] > peak_height):
        peak_height = tp0_y[i]
        peak_index = i
if (peak_index == -1):
    print("ERROR! Unable to determine peak time.")
    exit()
peak_time = tp0_t[peak_index]
print("Peak time: ", peak_time, "s")
print("Peak height: ", peak_height, "Au")


#   -- Cuff-to-Peak --
print("Cuff-Peak time:", peak_time - cuff_time, "s")
cuff_peak_indices = search_range(tp1_t, cuff_time + ((peak_time - cuff_time) / 2.0), (peak_time - cuff_time) / 2.0)
cuff_peak_auc = 0.0
for i in range(cuff_peak_indices[0], cuff_peak_indices[-1] - 1):
    cuff_peak_auc += ((tp1_t[i + 1] - tp1_t[i]) + ((tp1_y[i + 1] + tp1_y[i]) / 2.0))
print("Cuff-Peak AUC:", cuff_peak_auc, "Au*s")


#   -- Biological-Zero --
bio_zero_indices = search_range(tp1_t, cuff_time - (bio_zero_delta / 2.0), bio_zero_delta / 2.0)
bio_zero_height = 0.0
for i in range(bio_zero_indices[0], bio_zero_indices[-1]):
    bio_zero_height += tp1_y[i]
bio_zero_height = bio_zero_height / len(bio_zero_indices)
print("Biological-Zero height: ", bio_zero_height, "Au")

#   -- Baseline --
baseline_indices = search_range(tp1_t, user_baseline_end_t - (baseline_delta / 2.0), baseline_delta / 2.0)
baseline_height = 0.0
for i in range(baseline_indices[0], baseline_indices[-1]):
    baseline_height += tp1_y[i]
baseline_height = baseline_height / len(baseline_indices)
print("Baseline height: ", baseline_height, "Au")


#   -- Cuff + 1min AUC --
cuff_plus_min_indices = search_range(tp0_t, cuff_time + 30.0, 30.0)
cuff_plus_auc = 0.0
for i in range(cuff_plus_min_indices[0], cuff_plus_min_indices[-1] - 1):
    cuff_plus_auc += ((tp0_t[i + 1] - tp0_t[i]) + ((tp0_y[i + 1] + tp0_y[i]) / 2.0))
print("Cuff +1 min AUC:", cuff_plus_auc, "Au*s")


#   -- Peak return to Baseline --
return_index = -1
return_time = -1
for i in range(peak_index, len(tp0_t)):
    if (tp0_y[i] <= baseline_height):
        return_index = i
        break

if (return_index == -1):
    int("Warn! Unable to return index.")
else:
    #   -- Return- Time --

    return_time = tp0_t[return_index]
    print("Return time:", return_time, "s")
    print("Return delta:", return_time - peak_time, "s")

    return_time_indices = search_range(tp0_t, peak_time + ((return_time - peak_time) / 2.0), (return_time - peak_time) / 2.0)
    return_auc = 0.0
    for i in range(return_time_indices[0], return_time_indices[-1] - 1):
        return_auc += ((tp0_t[i + 1] - tp0_t[i]) + ((tp0_y[i + 1] + tp0_y[i]) / 2.0))
    print("Return AUC:", return_auc, "Au*s")


if (show_plots):
    plt.plot(t, y, 'r-', label='raw')

    plt.plot(t, y1, 'm-', label='high')
    plt.plot(tp1_t, tp1_y, 'mo', label='high tps')

    plt.plot(t, y0, 'b-', label='low')
    plt.plot(tp0_t, tp0_y, 'bo', label='low tps')

    plt.axvspan(tp1_t[cuff_indices[0]], tp1_t[cuff_indices[-1]], facecolor='g', alpha=0.5)
    plt.axvline(x=cuff_time, color='k', linestyle='--')

    plt.axvspan(tp0_t[peak_indices[0]], tp0_t[peak_indices[-1]], facecolor='g', alpha=0.5)
    plt.axvline(x=peak_time, color='k', linestyle='--')
    plt.axhline(y=peak_height, color='k', linestyle='--')

    plt.axhline(y=bio_zero_height, color='k', linestyle='--')
    plt.axvspan(tp1_t[bio_zero_indices[0]], tp1_t[bio_zero_indices[-1]], facecolor='c', alpha=0.5)

    plt.axhline(y=baseline_height, color='k', linestyle='--')
    plt.axvspan(tp1_t[baseline_indices[0]], tp1_t[baseline_indices[-1]], facecolor='c', alpha=0.5)

    plt.axvline(x=return_time, color='k', linestyle='--')

    plt.legend()
    plt.show()


if (not os.path.isfile("append.dat")):
    f = open("append.dat", 'w')
    f.write("patient_no")
    f.write("\t")
    f.write("data_file")
    f.write("\t")
    f.write("cuff_time")
    f.write("\t")
    f.write("peak_time")
    f.write("\t")
    f.write("peak_height")
    f.write("\t")
    f.write("peak_cuff_delta")
    f.write("\t")
    f.write("cuff_peak_auc")
    f.write("\t")
    f.write("bio_zero_height")
    f.write("\t")
    f.write("baseline_height")
    f.write("\t")
    f.write("cuff_plus_auc")
    f.write("\t")
    f.write("return_time")
    f.write("\t")
    f.write("peak_return_delta")
    f.write("\t")
    f.write("return_auc")
    f.write("\n")


f = open("append.dat", 'a')
f.write(str(patient_no))
f.write("\t")
f.write(data_file)
f.write("\t")
f.write(str(cuff_time))
f.write("\t")
f.write(str(peak_time))
f.write("\t")
f.write(str(peak_height))
f.write("\t")
f.write(str(peak_time - cuff_time))
f.write("\t")
f.write(str(cuff_peak_auc))
f.write("\t")
f.write(str(bio_zero_height))
f.write("\t")
f.write(str(baseline_height))
f.write("\t")
f.write(str(cuff_plus_auc))
f.write("\t")
f.write(str(return_time))
f.write("\t")
f.write(str(return_time - peak_time))
f.write("\t")
f.write(str(return_auc))
f.write("\n")

f = open("output_" + str(patient_no) + ".dat", 'w')
f.write("patient_no")
f.write("\t")
f.write("data_file")
f.write("\t")
f.write("cuff_time")
f.write("\t")
f.write("peak_time")
f.write("\t")
f.write("peak_height")
f.write("\t")
f.write("peak_cuff_delta")
f.write("\t")
f.write("cuff_peak_auc")
f.write("\t")
f.write("bio_zero_height")
f.write("\t")
f.write("baseline_height")
f.write("\t")
f.write("cuff_plus_auc")
f.write("\t")
f.write("return_time")
f.write("\t")
f.write("peak_return_delta")
f.write("\t")
f.write("return_auc")
f.write("\n")

f.write(str(patient_no))
f.write("\t")
f.write(data_file)
f.write("\t")
f.write(str(cuff_time))
f.write("\t")
f.write(str(peak_time))
f.write("\t")
f.write(str(peak_height))
f.write("\t")
f.write(str(peak_time - cuff_time))
f.write("\t")
f.write(str(cuff_peak_auc))
f.write("\t")
f.write(str(bio_zero_height))
f.write("\t")
f.write(str(baseline_height))
f.write("\t")
f.write(str(cuff_plus_auc))
f.write("\t")
f.write(str(return_time))
f.write("\t")
f.write(str(return_time - peak_time))
f.write("\t")
f.write(str(return_auc))
f.write("\n")
