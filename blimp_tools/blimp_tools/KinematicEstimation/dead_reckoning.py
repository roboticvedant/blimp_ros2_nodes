import numpy as np

class DeadReckoning:
    def __init__(self):
        # Initialize position and velocity
        self.position = np.array([0.0, 0.0, 0.0])  # Starting position
        self.velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
        self.prev_time = None

    def update(self, linear_acceleration, timestamp):
        """Update position and velocity using linear acceleration and time."""
        if self.prev_time is None:
            self.prev_time = timestamp
            return self.position

        # Time delta
        dt = timestamp - self.prev_time
        self.prev_time = timestamp

        # Integrate acceleration to update velocity and position
        self.velocity += linear_acceleration * dt
        self.position += self.velocity * dt

        return self.position