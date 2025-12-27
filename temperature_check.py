#Assignment day 22nd OCtober 2025
# No 8: Temperature check
# Ask the user to enter the temperature in Celsius
temperature = float(input("Enter the temperature in °C(figure only): "))

# Check the temperature condition
if temperature > 30:
    print("It's hot! Turn on the AC!")
elif temperature < 10:
    print("It's cold! Turn on the heater!")
else:
    print("It's pleasant.")
