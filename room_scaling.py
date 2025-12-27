from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Data
room_numbers = np.array([[1], [2], [3], [4], [5]])
room_prices = np.array([[10000], [20000], [30000], [100000], [200000]])

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
room_num_scaled = minmax_scaler.fit_transform(room_numbers)
room_price_scaled = minmax_scaler.fit_transform(room_prices)

# Standardization
std_scaler = StandardScaler()
room_num_standard = std_scaler.fit_transform(room_numbers)
room_price_standard = std_scaler.fit_transform(room_prices)

print("Room Number MinMax:\n", room_num_scaled)
print("Room Price MinMax:\n", room_price_scaled)
