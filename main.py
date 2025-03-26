import pandas as pd
import numpy as np
import tqdm
import multiprocessing as mp
import warnings
import time
import matplotlib.pyplot as plt
import psutil
import threading
import functools
import os
from typing import Dict

warnings.filterwarnings("ignore")

# Get initial dataset

wd = r'C:/Users/KasparasRutkauskas/OneDrive - Valyuz UAB/Desktop/Private/VU/Big Data Analysis/aisdk-2024-08-04.csv'


def pulling_all_at_once(wd):
    main_columns = ['# Timestamp', 'MMSI', 'Latitude', 'Longitude']
    dtst = pd.read_csv(
        wd,
        usecols=main_columns,
        dtype={'MMSI': 'int64', 'Latitude': 'float64', 'Longitude': 'float64'},
        parse_dates=['# Timestamp']
    )
    mmsi_values = set(dtst['MMSI'])
    # Removing rows that show the same info
    und_dtst = dtst.drop_duplicates(subset=main_columns)
    # Removing ships that only had one reading
    mask = und_dtst.MMSI.isin(und_dtst.MMSI.value_counts()[und_dtst.MMSI.value_counts() > 1].index)
    final_result = und_dtst[mask].sort_values('# Timestamp', ascending=True)

    return final_result, mmsi_values

# df, mmsi_values = pulling_all_at_once(wd)
# df.to_parquet('parquet_file', engine='pyarrow')

df = pd.read_parquet('parquet_file')

grouped_mmsi = list(df.groupby('MMSI'))

# Task 3A: Detect sudden, unrealistic location jumps that deviate significantly from expected vessel movement patterns.

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    bearing_degrees = (np.degrees(c) + 360) % 360
    return R * c, bearing_degrees

def calculate_speed(duration, distance):
    speed = distance / duration
    return speed

def direction_outlier_detection(v_data: pd.DataFrame) -> pd.DataFrame:

    outlier_columns = ['Outlier Type', 'MMSI', '# Timestamp', 'Distance tracked', 'Direction difference', 'Speed difference']

    v_data['Direction difference'] = v_data['direction'] - v_data['direction_lag3']
    v_data['Outlier Type'] = 'Sudden Direction Change'

    return v_data[(v_data['Direction difference'] > 0.001) & (v_data['Distance tracked'] != 0)][outlier_columns]
    


def sudden_location_outlier_detection(v_data: pd.DataFrame) -> pd.DataFrame:

    outlier_columns = ['Outlier Type', 'MMSI', '# Timestamp', 'Distance tracked', 'Direction difference', 'Speed difference']

    threshold = abs(v_data['Speed difference']).quantile(0.95)
    # print(threshold)
    outliers = v_data[(abs(v_data['Speed difference']) > threshold) & (v_data['Distance tracked'] != 0)]

    outliers['Outlier Type'] = 'Sudden Location Outlier'

    return outliers[outlier_columns]



def location_outlier_detection(chunk: pd.DataFrame) -> pd.DataFrame:

    """
    Detecting location outliers of these three scenarios:
        1. Sudden movement of location without in 0 time. 
        2. Sudden change of movement direction that is at least 0.01 departure from the average in the last 3 movements.
        3. Sudden change of location that is not in line with the average speed that was portrayed in the previous movement.
    """

    outlier_columns = ['Outlier Type', 'MMSI', '# Timestamp', 'Distance tracked', 'Direction difference', 'Speed difference']
    outliers = pd.DataFrame([], columns = outlier_columns)
    vessel_dt = chunk[1].copy()

    if len(vessel_dt) < 2:
        return None

    vessel_dt['previous_latitude'] = vessel_dt['Latitude'].shift(1)
    vessel_dt['previous_longitude'] = vessel_dt['Longitude'].shift(1)
    vessel_dt['previous_timestamp'] = vessel_dt['# Timestamp'].shift(1)

    distance, direction = calculate_distance(vessel_dt['previous_latitude'],
                                             vessel_dt['previous_longitude'],
                                             vessel_dt['Latitude'],
                                             vessel_dt['Longitude'])
    
    duration = (vessel_dt['# Timestamp'] - vessel_dt['previous_timestamp']).dt.total_seconds() / 60

    vessel_dt['Distance tracked'] = distance
    vessel_dt['direction'] = direction
    vessel_dt['duration'] = duration
    vessel_dt['speed (m/min)'] = calculate_speed(duration, distance)
    vessel_dt.loc[(vessel_dt['duration'] == 0) & (vessel_dt['Distance tracked'] == 0)]['speed (m/min)'] = 0
    vessel_dt.loc[(vessel_dt['duration'] == 0) & (vessel_dt['Distance tracked'] > 0)]['speed (m/min)'] = None
    vessel_dt['Speed difference'] = vessel_dt['speed (m/min)'] - vessel_dt['speed (m/min)'].shift(1)

    # Teleportation
    detected_outliers = vessel_dt[(vessel_dt['duration'] == 0) & (vessel_dt['Distance tracked'] > 0)]
    detected_outliers['Outlier Type'] = 'Teleportation'
    detected_outliers['Direction difference'] = None
    
    outliers = pd.concat([outliers, detected_outliers[outlier_columns]], ignore_index=True)

    # Drastic Direction changes
    vessel_dt['direction_lag3'] = vessel_dt['direction'].rolling(window=3).mean()

    outliers = pd.concat([outliers, direction_outlier_detection(vessel_dt)], ignore_index=True)

    # Location jumps will be detected by sudden speed changes:
    outliers = pd.concat([outliers, sudden_location_outlier_detection(vessel_dt)], ignore_index=True)
    
    return outliers



memory_usages = {}
memory_usage_perc = {}
cpu_usage_perc = {}

def track_max_memory(interval=0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            mem_usage = []
            mem_usage_perc = []
            cpu_perc = []

            def monitor():
                while not stop_event.is_set():
                    mem = process.memory_info().rss / (1024 ** 2)  # in MB
                    mem_percent = process.memory_percent()
                    cpu_percent = process.cpu_percent(interval=None)

                    mem_usage.append(mem)
                    mem_usage_perc.append(mem_percent)
                    cpu_perc.append(cpu_percent)
                    time.sleep(interval)

            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                monitor_thread.join()
            
            

            max_mem = max(mem_usage) if mem_usage else 0
            if func.__name__ == 'run_parallel':
                memory_usage_perc[f'{args[1]} processors'] = mem_usage_perc
                cpu_usage_perc[f'{args[1]} processors'] = cpu_perc
                memory_usages[f'{args[1]} processors'] = max_mem
                print(f"[{func.__name__} ({args[1]} cores)] Max Memory Usage: {max_mem:.2f} MB")
            else:
                memory_usage_perc['Serial'] = mem_usage_perc
                cpu_usage_perc['Serial'] = cpu_perc
                memory_usages['Serial'] = max_mem
                print(f"[{func.__name__}] Max Memory Usage: {max_mem:.2f} MB")
            return result, max_mem

        return wrapper
    return decorator


# sequential run
@track_max_memory(interval=0.15)
def run_sequential(grouped_mmsi : pd.DataFrame) -> pd.DataFrame:
    """
    Running the function sequentially 
    """

    all_outliers = []
    for grouping in tqdm.tqdm(grouped_mmsi, desc = 'Sequential run'):
        outliers_detected = location_outlier_detection(grouping)
        all_outliers.extend(outliers_detected)
    return all_outliers

# parallel run
@track_max_memory(interval=0.15)
def run_parallel(grouped_mmsi: pd.DataFrame, num_processors : int) -> pd.DataFrame:
    """
    Multiprocessing a function
    """

    with mp.Pool(processes=num_processors) as pool:
        # results = list(pool.imap(location_outlier_detection, grouped_mmsi))
        results = list(pool.imap_unordered(location_outlier_detection, grouped_mmsi))
    
    result_list = tqdm.tqdm([res for res in results if res is not None and not res.empty], desc = f'Parallel run {num_processors} processors')

    pd.concat(result_list, ignore_index=True).to_parquet('GPS_Spoofing_Results')



def performance_testing(runtimes : Dict) -> None:
    """
    Speed up performance results: Serial runtime / multiprocessing runtime
    """
    speedups = {}

    # Speed up testing
    print(f"Sequential Time: {runtimes['Serial']:.2f}s")
    for ptime in [*runtimes.keys()][1:]:
        speedup = runtimes['Serial'] / runtimes[ptime]
        print(f"Parallel Time ({ptime} processes): {runtimes[ptime]:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        speedups[ptime] = speedup
    return speedups


# Different configurations of parallelizing
cores = [2, 4, 6, 8]

runtimes = {}

def graphs(runtimes : Dict, memory_usages : Dict, speedups : Dict) -> None:
    """
    Creates two graphs based on the performance of the runs:
    1. Memory usage against run time for different configurations
    2. Memory usage over group for different configurations
    3. Speedups for different parallelization configurations
    """

    os.makedirs("Graphs", exist_ok=True)

    # Graph 1 

    names_rt, durations_rt = [*runtimes.keys()], [*runtimes.values()]
    _, mems_rt = [*memory_usages.keys()], [*memory_usages.values()]

    x = np.arange(start = 1, stop = len(names_rt)+1)

    width = 0.2
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting serial execution times
    bars1 = ax1.bar(x - width/2, durations_rt, width, label='Time (s)', color='skyblue')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_xlabel('Process')
    ax1.set_xticklabels(['Serial', '2 processors', '4 processors', '6 processors', '8 processors'])

    # Add bar labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

    ax2 = ax1.twinx()

    # Second bar plot (right y-axis)
    bars2 = ax2.bar(x + width/2, mems_rt, width, label='Memory (MB)', color='salmon')
    ax2.set_ylabel('Memory Usage (MB)')

    # Add bar labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0f}', ha='center', va='bottom')

    plt.title("Process Time vs Memory Usage")
    fig.tight_layout()

    plt.savefig("Graphs/process_times_v_memory_usage.png", dpi=300, bbox_inches='tight')

    # Graph 2s

    for comp in cpu_usage_perc.keys():
        print(comp)

        memory_pct, cpu_pct = memory_usage_perc[comp], cpu_usage_perc[comp]

        time_points = range(len(cpu_usage_perc[comp])) 

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Memory and CPU usage lines
        plt.plot(time_points, cpu_pct, label='CPU (%)', linewidth=2)
        plt.plot(time_points, memory_pct, label='Memory (%)', linestyle='--', linewidth=2)

        plt.xlabel('Time (intervals)')
        plt.ylabel('Usage')
        plt.title(f'Memory and CPU Usage Over Time {comp}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Graphs/memory_chart_{comp}.png", dpi=300, bbox_inches='tight')
    
    # Graph 3
    plt.figure(figsize=(8, 6))
    bars = plt.bar(speedups.keys(), speedups.values(), color='mediumseagreen')
    for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}x', ha='center', va='bottom')

    plt.xlabel('Configurations')
    plt.ylabel('Speedup vs sequential run')
    plt.title('Speedups for configurations')
    plt.savefig(f"Speedupchart.png", dpi=300, bbox_inches='tight')

    



# For testing start:

# runtimes = {'Serial' : 30,
#             '2 processors': 40,
#             '4 processors': 50,
#             '6 processors': 60,
#             '8 processors': 70}

# memory_usages = {'Serial' : 1800,
#                  '2 processors': 1744.734375,
#                  '4 processors': 1670.75,
#                  '6 processors': 735.1328125,
#                  '8 processors': 708.34765625}

# speedups = {'2 processors': 0.63,
#             '4 processors': 0.60,
#             '6 processors': 1.69,
#             '8 processors': 1.10}

# For testing stop


if __name__ == "__main__":

    time_init = time.time()
    run_sequential(grouped_mmsi)
    time_end = time.time()
    runtimes['Serial'] = time_end - time_init
    print(f'Serial execution took {(time_end - time_init):.2f} seconds.')

    for corenum in cores:
        time_init = time.time()
        run_parallel(grouped_mmsi, corenum)
        time_end = time.time()
        # memory_after = memory_before = process.memory_info().rss / (1024 ** 2)
        
        runtimes[f'{corenum} processors'] = time_end - time_init
        
        print(f'Multithreading with {corenum} processes took {(time_end - time_init):.2f} seconds.')

    speedups = performance_testing(runtimes)
    # print(memory_usages)
    graphs(runtimes, memory_usages, speedups)


# import pandas as pd
# pd.read_parquet('GPS_Spoofing_Results')