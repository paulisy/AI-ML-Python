#Assignment day 22nd OCtober 2025
# No 2: Weather Station
# Enter the temperature in Celsius
celsius = float(input("Enter temperature in Celsius(figure only): "))

# Convert Celsius to Fahrenheit
fahrenheit = (celsius * 9/5) + 32

# Calculate how many full 5-degree increments are in the Celsius value
increments = int(celsius // 5)
remainder = int(celsius % 5)

# Display results
print('----------------------------------------------')
print(f"\nTemperature in Fahrenheit: {fahrenheit:.2f} °F")
print(f"Full 5-degree increments in Celsius: {increments}")
print(f"Remaining degrees: {remainder}")
