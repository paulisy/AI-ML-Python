import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -------------------------------
# Original Data
# -------------------------------
room_numbers = np.array([[1], [2], [3], [4], [5]])
room_prices = np.array([[10000], [20000], [30000], [100000], [200000]])

# -------------------------------
# Scaling
# -------------------------------
# Min-Max Scaling
minmax_scaler = MinMaxScaler()
room_num_minmax = minmax_scaler.fit_transform(room_numbers)
room_price_minmax = minmax_scaler.fit_transform(room_prices)

# Standardization
std_scaler = StandardScaler()
room_num_std = std_scaler.fit_transform(room_numbers)
room_price_std = std_scaler.fit_transform(room_prices)

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(12, 6))

# ---- Plot 1: Room Numbers ----
plt.subplot(1, 2, 1)
plt.plot(room_numbers, room_numbers, 'o-', label='Original', linewidth=2)
plt.plot(room_numbers, room_num_minmax, 'o--', label='Min–Max Scaled')
plt.plot(room_numbers, room_num_std, 'o--', label='Standardized')
plt.title('Room Number Scaling')
plt.xlabel('Room Number')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)

# ---- Plot 2: Room Prices ----
plt.subplot(1, 2, 2)
plt.plot(room_numbers, room_prices, 'o-', label='Original (₦)', linewidth=2)
plt.plot(room_numbers, room_price_minmax, 'o--', label='Min–Max Scaled')
plt.plot(room_numbers, room_price_std, 'o--', label='Standardized')
plt.title('Room Price Scaling')
plt.xlabel('Room Number')
plt.ylabel('Price / Scaled Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
