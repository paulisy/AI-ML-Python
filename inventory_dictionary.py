#Assignment day 22nd OCtober 2025
# No 7: Inventory
# Store inventory
inventory = {'apples': 10, 'bananas': 20, 'oranges': 15}

# Display initial inventory
print("🛒 Welcome to the Fruit Store!")
print("Current Inventory:", inventory)
print("-----------------------------------")

# input the number of fruits you want
apples_bought = int(input("How many apples would you like to buy? "))
bananas_bought = int(input("How many bananas would you like to buy? "))
oranges_bought = int(input("How many oranges would you like to buy? "))

# Check and update inventory
if apples_bought <= inventory["apples"] and bananas_bought <= inventory["bananas"] and oranges_bought <= inventory["oranges"]:
    inventory["apples"] -= apples_bought
    inventory["bananas"] -= bananas_bought
    inventory["oranges"] -= oranges_bought
    print("\n✅ Purchase successful!")
else:
    print("\n❌ Sorry, not enough stock available for one or more items.")

# Display updated inventory
print("-----------------------------------")
print("Updated Inventory:", inventory)
