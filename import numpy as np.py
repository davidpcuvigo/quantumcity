import numpy as np
import matplotlib.pyplot as plt
from network import HorizontalChannel, DownlinkChannel, UplinkChannel
import cn2
# Example parameters
# Define the channel length in km
length_km_50 = 100  # 50 km Channel length
length_km_100 = length_km_50*2  # 100 km Channel length
array_size=int(1e3)

eta_ch = np.linspace(1e-7, 1, array_size)  # Transmission efficiency values

# Cn2 for the horizontal channel (you can adjust this if needed)
Cn2_horizontal = cn2.hufnagel_valley(135000, 10.0, 9.6e-14)

downchannel = DownlinkChannel(W0=0.1, #converted to m
                                                            rx_aperture=0.3, #converted to m
                                                            obs_ratio=0.3,
                                                            n_max=150,
                                                            Cn0=9.6e-14,
                                                            wind_speed=10,
                                                            wavelength=1e-9*1550, #converted to m
                                                            ground_station_alt=0.02, #converted to km
                                                            aerial_platform_alt=35,
                                                            zenith_angle=0,
                                                            pointing_error=0,
                                                            tracking_efficiency=1,
                                                            Tatm=1,
                                                            integral_gain=1,
                                                            control_delay=2e-3,
                                                            integration_time=1e-3,
                                                            )
downlink_errors=downchannel._compute_loss_probability(length=35,n_samples=array_size)
 
upchannel = UplinkChannel(D_tx=0.4,
                                                            R_rx=0.4, 
                                                            obs_ratio=0.3,
                                                            n_max=150,
                                                            Cn0=9.6e-14,
                                                            wind_speed=10,
                                                            wavelength=1e-9*1550, #converted to m
                                                            ground_station_alt=0.02, #converted to km
                                                            aerial_platform_alt=35,
                                                            zenith_angle=0,
                                                            pointing_error=0,
                                                            tracking_efficiency=1,
                                                            Tatm=1,
                                                            integral_gain=1,
                                                            control_delay=2e-3,
                                                            integration_time=1e-3,
                                                            )
uplink_errors=upchannel._compute_loss_probability(length=35,n_samples=array_size)
            
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
plt.scatter(eta_ch, downlink_errors, label="Downlink Loss probability", color="r", s=0.1)

# Plot the multiplication of two 50 km channels
plt.scatter(eta_ch, uplink_errors, label="Uplink loss probability", color="g", s=0.1)

# Add labels and title
plt.xlabel("Iteration")
plt.ylabel("Loss probability")
plt.title("Photon loss probabilit for uplink and downlink channels")
plt.ylim(0,1)
# Add a legend to distinguish the curves
plt.legend()

# Add gridlines for better readability
plt.grid()

# Save the combined plot
plt.savefig("combined_channel_pdf_plot.png")
