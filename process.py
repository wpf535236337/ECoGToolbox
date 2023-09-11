import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import random


def denoise():
    # band-stop filter 0~90 Hz
    # notch filter 60HZ
    pass


def selectchannels():
    # may based on psd
    pass


def ecog_resample(ecog_data, original_sampling_rate, target_sampling_rate):
    # resampled_data = resample(ecog_data, int(len(ecog_data) * (target_sampling_rate / original_sampling_rate)))
    # directly using slice to speed up
    resampled_data = ecog_data[:, ::int(
        original_sampling_rate/target_sampling_rate)]
    return resampled_data


def segment_with_overlap(ecog_data, segment_length, overlap_ratio=0):
    """
    Segment ECoG data into overlapping fixed-size segments using NumPy.

    Args:
        ecog_data (numpy.ndarray): The ECoG data in the shape (num_channels, num_samples).
        segment_length (int): The desired length of each segment in number of samples.
        overlap_ratio (float): The overlap ratio between segments (0.0 to 1.0).

    Returns:
        List[numpy.ndarray]: A list of segmented ECoG data, where each element is a segment.
    """
    _, num_samples = ecog_data.shape

    # Calculate the overlap amount based on the overlap ratio
    overlap_amount = int(segment_length * overlap_ratio)

    # Initialize a list to store segments
    segments = []

    # Start creating segments with overlap
    start_idx = 0
    while start_idx <= (num_samples-segment_length):
        end_idx = start_idx + segment_length

        # Extract the segment and add it to the list
        segment_data = ecog_data[:, start_idx:end_idx]
        segments.append(segment_data)

        # Move the start index for the next segment with overlap
        start_idx += segment_length - overlap_amount

    return segments


def extend_array(original_array, desired_length):
    """
    Extend an array using interpolation.

    Args:
        original_array (numpy.ndarray): The original array to be extended.
        desired_length (int): The desired length of the extended array.
        method (str): The interpolation method ('linear', 'nearest', 'cubic', etc.).
    Returns:
        numpy.ndarray: The extended array.
    # Example usage:
        original_array = np.array([1, 2, 3, 4])
        desired_length = 10
        extended_array = extend_array(original_array, desired_length, method='linear')
        print("Original array:", original_array)
        print("Extended array:", extended_array)
    """
    # Create an array of indices for the original array
    indices = np.arange(len(original_array))

    # Create an array of indices for the extended array
    extended_indices = np.linspace(0, len(original_array) - 1, desired_length)

    # Use interpolation to extend the original array
    extended_array = np.interp(extended_indices, indices, original_array)

    # Binary
    extended_array[extended_array < 0.5] = 0
    extended_array[extended_array >= 0.5] = 1

    return extended_array


def split_dataset(X, y, val_rate=0.2):
    # train-test 13:7
    # Assuming you have a dataset X (features) and y (labels/targets)
    split_num = int(len(y)*(13.0)/20)  # as claim
    X_temp, y_temp, X_test, y_test = X[:split_num], y[:split_num], X[split_num:], y[split_num:]
    # Split the dataset into training (80%), validation (%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rate, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


def rebalance(data, gt):
    # Calculate the number of samples you want to keep for each class
    max_samples_per_class = max(gt.value_counts())

    # Split the dataset into separate subsets for each class
    class_0_samples = data[gt == 0]
    class_1_samples = data[gt == 1]

    # Oversample the smaller class to match the number of samples in the larger class
    oversampled_class_1 = class_1_samples.sample(
        n=max_samples_per_class, replace=True, random_state=42)

    # Combine the oversampled class and the larger class into a single dataset
    oversampled_data = pd.concat([oversampled_class_1, class_0_samples])
    return oversampled_data


def save_dataset(file_path, x, y):
    # np.savez(file_path, x=x, y=y)
    with h5py.File(file_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('x', data=x)
        hdf5_file.create_dataset('y', data=y)


def save_slices(save_root, data_x, label_y):
    class_name = {0.0:'fail', 1.0: 'success'}
    for i, (slice_data, slice_y) in enumerate(zip(data_x, label_y)):
        # Create a zero-padded filename
        filename = f"slice_{i:08d}.npy"
        
        # Define the full path for saving
        save_class_path = os.path.join(save_root, class_name[slice_y])
        os.makedirs(save_class_path, exist_ok=True)
        
        # Save the slice as a .npy file
        np.save(os.path.join(save_class_path, filename), slice_data)


def scan_all_files(directory, extension):
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))

    return file_list


def save_txt(save_root, stage, resample=True):

    data_paths = []
    
    x_0 = scan_all_files(os.path.join(save_root, stage, 'fail'), '.npy')
    x_1 = scan_all_files(os.path.join(save_root, stage, 'success'), '.npy')

    data_paths.extend(x_0)

    if resample:
        data_paths.extend(x_1*int(len(x_0)/len(x_1)))
        print('source', len(x_0), len(x_1))
        print('resample', len(x_0), len(data_paths) - len(x_0))
        flag_resample = 'resample'
    else:
        data_paths.extend(x_1)
        print('source', len(x_0), len(x_1))
        flag_resample = 'not_resample'
    
    with open(os.path.join(save_root, '{}_{}_paths.txt'.format(stage, flag_resample)), 'w') as data_file:
        data_file.write('\n'.join(data_paths))


def preprocess_ecog_dataset(mat_data_path, csv_file_path, save_root, target_sampling_rate=4000, segment_length=100, val_rate=0.2):
    # Load ECoG data and corresponding labels
    raw_data = scio.loadmat(mat_data_path)
    ecog_data = np.array(raw_data['nx_raw_data']).T
    df = pd.read_csv(csv_file_path)
    gt = df['Successful']

    # Resample ECoG data
    resampled_data = ecog_resample(ecog_data, original_sampling_rate=4000, target_sampling_rate=target_sampling_rate)

    # Segment ECoG data
    segment_data = segment_with_overlap(resampled_data, segment_length=segment_length, overlap_ratio=0)

    # print(len(gt), len(segment_data))

    # Extend labels to match the segmented data
    algn_gt = extend_array(gt, len(segment_data))

    # Split the dataset into train, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(segment_data, algn_gt, val_rate=val_rate)

    # Define the directory for saving slices and file paths
    save_root = os.path.join(save_root, "segment_not_downsample")

    # Save slices as .npy files
    save_slices(os.path.join(save_root, 'train'), X_train, y_train)
    save_slices(os.path.join(save_root, 'val'), X_val, y_val)
    save_slices(os.path.join(save_root, 'test'), X_test, y_test)

    # Save file paths to .txt files
    save_txt(save_root, 'train')
    save_txt(save_root, 'val')
    save_txt(save_root, 'test')
    
    # not balance class
    save_txt(save_root, 'train', resample=False)
    save_txt(save_root, 'val', resample=False)
    save_txt(save_root, 'test', resample=False)


if __name__ == '__main__':
    data_root = ""# dataset 
    mat_path = os.path.join(data_root, "raw_dataset_path")
    csv_file_path = os.path.join(data_root, "label_dataset_path")
    save_root = ""
    preprocess_ecog_dataset(mat_path, csv_file_path, save_root)

