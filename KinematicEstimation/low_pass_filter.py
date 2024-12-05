import numpy as np
from collections import deque

class LowPassFilter:
    def __init__(self, alpha, median_window):
        """
        Initializes a low-pass filter with an optional moving median filter.

        Args:
            alpha (float): The smoothing factor for the low-pass filter (0 < alpha <= 1).
            median_window (int): The window size for the moving median filter.
        """
        self.alpha = alpha
        self.previous = None
        self.median_window = median_window
        self.median_buffer = deque(maxlen=median_window)

    def apply(self, current):
        """
        Applies the moving median filter and then the low-pass filter to the current data.

        Args:
            current (float): The current value to be filtered.

        Returns:
            float: The filtered value.
        """
        # Add the current value to the median buffer
        self.median_buffer.append(current)

        # Apply the moving median filter
        if len(self.median_buffer) == self.median_window:
            median_filtered = np.median(self.median_buffer)
        else:
            # Use the raw value if the buffer is not full yet
            median_filtered = current

        # Apply the low-pass filter
        if self.previous is None:
            # Initialize with the first value
            self.previous = median_filtered
        filtered = self.alpha * median_filtered + (1 - self.alpha) * self.previous
        self.previous = filtered

        return filtered

class VectorLowPassFilter:
    def __init__(self, alpha=0.1, median_window=1):
        """
        Initializes low-pass filters for 3D vector data (e.g., X, Y, Z).

        Args:
            alpha (float): Smoothing factor for the low-pass filter.
        """
        self.filter_x = LowPassFilter(alpha, median_window)
        self.filter_y = LowPassFilter(alpha, median_window)
        self.filter_z = LowPassFilter(alpha, median_window)

    def filter(self, vector_data):
        """
        Applies low-pass filtering to 3D vector data.

        Args:
            vector_data (dict): A dictionary with 'x', 'y', 'z' keys.

        Returns:
            dict: Filtered 3D vector data.
        """
        if isinstance(vector_data, dict):
            filtered_data = {
                'x': self.filter_x.apply(vector_data['x']),
                'y': self.filter_y.apply(vector_data['y']),
                'z': self.filter_z.apply(vector_data['z']),
            }
        elif isinstance(vector_data, np.ndarray):
            filtered_data = {
                'x': self.filter_x.apply(vector_data[0]),
                'y': self.filter_y.apply(vector_data[1]),
                'z': self.filter_z.apply(vector_data[2]),
            }
        else:
            raise TypeError("vector_data must be a dictionary with 'x', 'y', 'z' keys or a numpy array.")
        
        return filtered_data



class SensorDataFilter:
    def __init__(self, alpha_lin_acc=0.3, alpha_ang_vel=0.2, alpha_mag=0.3):
        """
        Initializes low-pass filters for magnetometer, linear acceleration,
        and angular velocity data.

        Args:
            alpha (float): Smoothing factor for the low-pass filter.
        """
        self.magnetometer_filter = VectorLowPassFilter(alpha_mag, median_window=10)
        self.linear_acceleration_filter = VectorLowPassFilter(alpha_lin_acc)
        self.angular_velocity_filter = VectorLowPassFilter(alpha_ang_vel)

    def filter_magnetometer(self, magnetometer_data):
        """
        Filters magnetometer data.

        Args:
            magnetometer_data (dict): Magnetometer data with 'x', 'y', 'z' keys.

        Returns:
            dict: Filtered magnetometer data.
        """
        return self.magnetometer_filter.filter(magnetometer_data)

    def filter_linear_acceleration(self, linear_acceleration_data):
        """
        Filters linear acceleration data.

        Args:
            linear_acceleration_data (dict): Linear acceleration data with 'x', 'y', 'z' keys.

        Returns:
            dict: Filtered linear acceleration data.
        """
        return self.linear_acceleration_filter.filter(linear_acceleration_data)

    def filter_angular_velocity(self, angular_velocity_data):
        """
        Filters angular velocity data.

        Args:
            angular_velocity_data (dict): Angular velocity data with 'x', 'y', 'z' keys.

        Returns:
            dict: Filtered angular velocity data.
        """
        return self.angular_velocity_filter.filter(angular_velocity_data)
