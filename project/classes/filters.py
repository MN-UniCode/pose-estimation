from scipy.signal import butter, sosfilt_zi, sosfilt, savgol_coeffs
from collections import deque
import numpy as np


class Butterworth:
    def __init__(self, order, cutoff, btype='lowpass', fs=1.0):
        """
        Initialize the Butterworth filter in SOS (Second-Order Sections) format
        with an internal state buffer for real-time processing.

        Parameters:
        - order: int, filter order.
        - cutoff: float or list of floats, cutoff frequency/frequencies.
        - btype: str, filter type ('lowpass', 'highpass', 'bandpass', 'bandstop').
        - fs: float, sampling frequency.
        """
        # Design the filter using SOS output for numerical stability
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

        # Initialize filter state (zi) for sosfilt, one state per section
        self.zi = sosfilt_zi(self.sos)
        self.zi = np.zeros_like(self.zi)  # Initialize buffer with zeros

    def filter(self, sample):
        """
        Filter a single scalar sample, updating the internal filter state.

        Parameters:
        - sample: float, input scalar sample.

        Returns:
        - filtered_sample: float, filtered output sample.
        """
        # Reshape sample as an array of length 1 (required by scipy's sosfilt)
        x = np.array([sample])

        # Apply filter and update state
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y[0]


class ButterworthMultichannel:
    """
    Vectorized Butterworth filter with independent state buffers for multiple channels.

    Each channel is filtered independently but shares the same SOS coefficients.
    Suitable for streaming (sample-by-sample) or block updates.
    """

    def __init__(self, n_channels, order, cutoff, btype='lowpass', fs=1.0):
        # Design one Butterworth filter in SOS form
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

        # Create state buffers: One zi buffer per SOS section per channel
        # zi_template shape: (n_sections, 2)
        zi_template = sosfilt_zi(self.sos)
        n_sections, n_zi = zi_template.shape

        # State shape: (n_channels, n_sections, 2)
        self.zi = np.zeros((n_channels, n_sections, n_zi))

        self.n_channels = n_channels
        self.n_sections = n_sections

    def filter(self, x):
        """
        Filter one sample per channel (streaming mode).

        Parameters
        ----------
        x : array_like, shape (n_channels,)
            Current input sample for all channels.

        Returns
        -------
        y : ndarray, shape (n_channels,)
            Filtered output for all channels.
        """
        x = np.asarray(x)
        y = x.copy()

        # Manually apply SOS sections (Direct-form II transposed) across all channels
        for s, (b0, b1, b2, a0, a1, a2) in enumerate(self.sos):
            y_new = b0 * y + self.zi[:, s, 0]

            # Update state variables
            self.zi[:, s, 0] = b1 * y - a1 * y_new + self.zi[:, s, 1]
            self.zi[:, s, 1] = b2 * y - a2 * y_new

            y = y_new

        return y


class SavitzkyGolay:
    def __init__(self, window_length, polyorder, deriv=0, delta=1.0):
        """
        Initialize a Single-channel Savitzky-Golay filter.

        Parameters:
        - window_length: odd int, length of the filter window (number of coefficients).
        - polyorder: int, order of the polynomial to fit.
        - deriv: int, order of the derivative to compute (0 means smoothing).
        - delta: float, sample spacing (used only if deriv > 0).
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")

        self.window_length = window_length
        self.half_window = window_length // 2
        self.coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
        self.buffer = [0.0] * window_length  # Circular buffer for streaming input

    def filter(self, sample):
        """
        Filter a single scalar sample using the Savitzky-Golay filter.

        Parameters:
        - sample: float, input scalar sample.

        Returns:
        - filtered_sample: float, filtered output.
        """
        # Shift buffer and append new sample
        self.buffer.pop(0)
        self.buffer.append(sample)

        # Handle startup: if the buffer isn't logically full (conceptually), pad with zeros
        # Note: In this implementation, the buffer is pre-filled with 0.0, so this check
        # mainly handles the theoretical 'warm-up' logic.
        if len(self.buffer) < self.window_length:
            padded = [0.0] * (self.window_length - len(self.buffer)) + self.buffer
        else:
            padded = self.buffer

        # Apply coefficients (dot product)
        return float(np.dot(self.coeffs, padded))


class SavitzkyGolayMultichannel:
    def __init__(self, num_channels, window_length, polyorder, deriv=0, delta=1.0):
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")

        self.window_length = window_length
        self.num_channels = num_channels
        self.coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

        # Buffer shape: (window_length, num_channels)
        self.buffer = np.zeros((window_length, num_channels))
        self.filled = False

    def filter(self, samples):
        """
        Apply filter to current samples.

        Parameters:
        - samples: np.array of shape (num_channels,) containing flattened current data.
        """
        # Shift the buffer (scroll up)
        # Discard the oldest row, move everything up by one
        self.buffer[:-1] = self.buffer[1:]
        # Insert new samples at the end (newest row)
        self.buffer[-1] = samples

        # Startup phase management
        if not self.filled:
            # Optional: Fill buffer with the first sample to avoid initial jump/artifacts
            if np.all(self.buffer[0] == 0):
                self.buffer[:] = samples
            self.filled = True  # Simplified: consider full after first input or wait N frames

        # Vectorized dot product
        # coeffs shape: (window,)
        # buffer shape: (window, channels)
        # result shape: (channels,)
        return np.dot(self.coeffs, self.buffer)


class Hampel:
    def __init__(self, window_size=5, n_sigma=3, replace_with='mean'):
        """
        Real-time 1D Hampel filter with internal state.

        Args:
            window_size (int): Half-size of the window (buffer size = 2k + 1).
            n_sigma (float): Threshold in MAD units for outlier detection.
            replace_with (str): 'mean' or 'median' for outlier replacement.
        """
        self.k = window_size
        self.n_sigma = n_sigma
        self.replace_with = replace_with
        self.scale = 1.4826  # Scale factor for Gaussian noise consistency
        self.buffer = deque(maxlen=2 * self.k + 1)

    def filter(self, value):
        """
        Process a new sample and return the filtered result.

        Args:
            value (float): Incoming value from the time series.

        Returns:
            float: Filtered value (either original or corrected).
        """
        self.buffer.append(value)

        # Not enough data yet to make a decision
        if len(self.buffer) < self.buffer.maxlen:
            return value

        window = np.array(self.buffer)
        center_idx = self.k
        center_value = window[center_idx]

        # Calculate Median Absolute Deviation (MAD)
        median = np.median(window)
        mad = self.scale * np.median(np.abs(window - median))

        # If MAD is 0, the window is constant (no noise/variation), return center
        if mad == 0:
            return center_value

        # Check for outlier
        if abs(center_value - median) > self.n_sigma * mad:
            if self.replace_with == 'mean':
                return np.mean(window)
            else:
                return median

        return center_value


class HampelMultichannel:
    def __init__(self, num_channels, window_size=5, n_sigma=3, replace_with='median'):
        """
        Real-time Multichannel Hampel filter for landmark data (e.g., 99 channels).

        Args:
            num_channels (int): Number of channels (e.g., 33 * 3 = 99).
            window_size (int): Half-size of the window (total buffer size = 2k + 1).
            n_sigma (float): Threshold in MAD units for outlier detection.
            replace_with (str): 'mean' or 'median' for outlier replacement.
        """
        self.k = window_size
        self.n_sigma = n_sigma
        self.replace_with = replace_with
        self.scale = 1.4826  # Scale factor for Gaussian noise
        self.num_channels = num_channels
        self.buffer_size = 2 * self.k + 1

        # Initialize a buffer (deque) for each channel
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(num_channels)]

    def filter(self, values):
        """
        Process a new array of samples (one value per channel) and return filtered results.

        Args:
            values (np.array): 1D array of input values (size: num_channels).

        Returns:
            np.array: 1D array of filtered values (size: num_channels).
        """
        if values.shape[0] != self.num_channels:
            raise ValueError(f"Input array size must be {self.num_channels}, but got {values.shape[0]}.")

        filtered_values = np.empty_like(values, dtype=float)

        # Iterate through each channel
        for i in range(self.num_channels):
            value = values[i]
            current_buffer = self.buffers[i]
            current_buffer.append(value)

            # If buffer not full, return original value
            if len(current_buffer) < self.buffer_size:
                filtered_values[i] = value
                continue

            window = np.array(current_buffer)
            center_value = window[self.k]

            median = np.median(window)
            mad = self.scale * np.median(np.abs(window - median))

            threshold = 0.07  # Hardcoded threshold clamp

            if mad == 0:
                # Note: Logic here clips based on 'filtered_values[i]' which might be uninitialized.
                # Assuming intention is to clip 'center_value'.
                filtered_values[i] = np.clip(
                    center_value,
                    center_value - threshold,
                    center_value + threshold
                )
            elif np.abs(center_value - median) > self.n_sigma * mad:
                # Outlier detected
                if self.replace_with == 'mean':
                    filtered_values[i] = np.mean(window)
                else:
                    filtered_values[i] = median
            else:
                # Valid value, apply threshold clipping
                filtered_values[i] = np.clip(
                    center_value,
                    center_value - threshold,
                    center_value + threshold
                )

        return filtered_values