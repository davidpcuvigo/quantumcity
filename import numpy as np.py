import numpy as np
import matplotlib.pyplot as plt
from network import HorizontalChannel, DownlinkChannel
import cn2
# Example parameters
# Define the channel length in km
length_km_50 = 100  # 50 km Channel length
length_km_100 = length_km_50*2  # 100 km Channel length
array_size=int(1e3)

eta_ch = np.linspace(1e-7, 1, array_size)  # Transmission efficiency values

# Cn2 for the horizontal channel (you can adjust this if needed)
Cn2_horizontal = cn2.hufnagel_valley(135000, 10.0, 9.6e-14)

# HorizontalChannel class instantiation (assuming this class is defined)
horiz = HorizontalChannel(
    W0=0.4, 
    rx_aperture=0.4, 
    obs_ratio=0.3, 
    wavelength=1550 * 1e-9, 
    pointing_error=1e-6, 
    tracking_efficiency=0.85, 
    Cn2=Cn2_horizontal
)
               
# Compute PDFs for 50 km and 100 km channels
channel_pdf_50 = horiz._compute_loss_probability(length_km_50, n_samples=array_size)
channel_pdf_100 = horiz._compute_loss_probability(length_km_100, n_samples=array_size)

# Multiply two 50 km channels (element-wise multiplication)
channel_pdf_50_multiplied = channel_pdf_50

# Plot both on the same plot
plt.figure(figsize=(8, 5))

# Plot the 100 km channel PDF
plt.scatter(eta_ch, channel_pdf_100, label="100 km Channel PDF", color="r", s=0.1)

# Plot the multiplication of two 50 km channels
plt.scatter(eta_ch, channel_pdf_50_multiplied, label="Multiplication of Two 50 km Channels", color="g", s=0.1)

# Add labels and title
plt.xlabel("Channel Efficiency")
plt.ylabel("Probability Density")
plt.title("Probability Density Functions of Channel Efficiencies")
plt.ylim(0,1)
# Add a legend to distinguish the curves
plt.legend()

# Add gridlines for better readability
plt.grid()

# Save the combined plot
plt.savefig("combined_channel_pdf_plot.png")
