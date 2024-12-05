import numpy as np
import pandas as pd
from scipy.signal import TransferFunction, lsim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def display_transfer_function_latex(K, omega_n, zeta):
    """
    Display the transfer function in a beautifully formatted LaTeX-style plot.
    """
    # Prepare LaTeX string for the transfer function
    tf_latex = (
        r"$G(s) = \frac{" + f"{K:.4f} \\cdot {omega_n:.4f}^2" + r"}"
        r"{s^2 + 2 \cdot " + f"{zeta:.4f} \\cdot {omega_n:.4f}" + r" \cdot s + {omega_n:.4f}^2}$"
    )

    # Create a plot with no axes, just for displaying the equation
    plt.figure(figsize=(8, 2))
    plt.text(0.5, 0.5, tf_latex, fontsize=16, ha='center', va='center', wrap=True)
    plt.axis('off')  # Turn off the axes
    plt.title("Estimated Transfer Function", fontsize=18, pad=20)

    # Show the plot
    plt.show()

def resample_data(time, input_signal, output_signal, num_points=1000):
    """
    Resample data to ensure equally spaced time steps.
    """
    # Create a new equally spaced time array
    time_resampled = np.linspace(time[0], time[-1], num_points)

    # Interpolate the input and output signals
    input_interpolator = interp1d(time, input_signal, kind='linear')
    output_interpolator = interp1d(time, output_signal, kind='linear')

    input_resampled = input_interpolator(time_resampled)
    output_resampled = output_interpolator(time_resampled)

    return time_resampled, input_resampled, output_resampled

def read_data(file_name):
    """
    Read the CSV file and extract the input (angular velocity command) and output (yaw rate or angle).
    """
    try:
        data = pd.read_csv(file_name)
        if data.empty:
            raise ValueError("CSV file is empty or improperly formatted.")
        
        print("CSV Data Preview:")
        print(data.head())  # Print the first few rows for debugging
        
        time = data['Time'].to_numpy()
        input_signal = data['Angular_Z_Command'].to_numpy()
        output_signal = data['Yaw_Rate'].to_numpy()  # Replace with 'Yaw_Angle' if needed
        
        if len(time) == 0 or len(input_signal) == 0 or len(output_signal) == 0:
            raise ValueError("One or more columns are empty.")
        
        return time, input_signal, output_signal
    except Exception as e:
        print(f"Error reading data: {e}")
        exit(1)



def second_order_response(t, K, omega_n, zeta):
    """
    Step response of a second-order system.
    """
    # Compute the exponential decay term
    exp_term = np.exp(-zeta * omega_n * t)
    
    # Compute the sinusoidal term
    if zeta < 1:  # Underdamped
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        response = K * (1 - exp_term * (np.cos(omega_d * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t)))
    else:  # Critically or overdamped
        response = K * (1 - exp_term * (1 + zeta * t))
    
    return response


def estimate_second_order_tf(time, input_signal, output_signal):
    """
    Estimate a second-order transfer function G(s) = K*omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2).
    """
    # Normalize time to start from zero
    time = time - time[0]

    # Use curve_fit to estimate parameters
    popt, _ = curve_fit(second_order_response, time, output_signal, p0=[1.0, 1.0, 0.5])  # Initial guesses for K, omega_n, zeta

    K, omega_n, zeta = popt
    print(f"Estimated Transfer Function: G(s) = {K} * {omega_n}^2 / (s^2 + 2*{zeta}*{omega_n}*s + {omega_n}^2)")
    return K, omega_n, zeta


def simulate_second_order_tf(K, omega_n, zeta, time, input_signal):
    """
    Simulate the response of the identified second-order transfer function.
    """
    num = [K * omega_n**2]       # Numerator of transfer function
    den = [1, 2 * zeta * omega_n, omega_n**2]  # Denominator of transfer function
    system = TransferFunction(num, den)

    # Simulate the system response to the input signal
    _, y_sim, _ = lsim(system, U=input_signal, T=time)
    return y_sim


def plot_results(time, input_signal, output_signal, y_sim):
    """
    Plot the input, output, and simulated response.
    """
    plt.figure(figsize=(12, 6))

    # Plot input signal
    plt.subplot(2, 1, 1)
    plt.plot(time, input_signal, label='Input (Angular_Z_Command)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Input')
    plt.legend()
    plt.grid()

    # Plot output and simulated response
    plt.subplot(2, 1, 2)
    plt.plot(time, output_signal, label='Measured Output (Yaw_Rate)', color='green')
    plt.plot(time, y_sim, label='Simulated Output', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    # Path to the CSV file
    file_name = "data_2024-11-24_16-25-43.csv"  # Replace with your file name

    # Read data
    time, input_signal, output_signal = read_data(file_name)

    # Resample data to ensure equally spaced time steps
    time, input_signal, output_signal = resample_data(time, input_signal, output_signal)

    # Estimate second-order transfer function
    K, omega_n, zeta = estimate_second_order_tf(time, input_signal, output_signal)

    # Simulate the response of the identified system
    y_sim = simulate_second_order_tf(K, omega_n, zeta, time, input_signal)

    # Plot results
    plot_results(time, input_signal, output_signal, y_sim)

    # Display the transfer function in LaTeX-style
    display_transfer_function_latex(K, omega_n, zeta)



if __name__ == "__main__":
    main()
