#Assignment day 22nd OCtober 2025
# No 9: If Statements – Discount Eligibility with Multiple Conditions
# Define customer type and purchase amount
# customer type can be "VIP", "Regular", or "New"
customer_type = "Regular"      
purchase_amount = 170      # amount in dollars

# Initialize discount
discount = 0

# Check customer type and apply discount rules
if customer_type == "VIP":
    if purchase_amount >= 100:
        discount = 0.20 * purchase_amount
    else:
        discount = 0.10 * purchase_amount
elif customer_type == "Regular":
    if purchase_amount >= 150:
        discount = 0.15 * purchase_amount
    else:
        discount = 0
elif customer_type == "New":
    discount = 0
else:
    print("Invalid customer type")

# Calculate final price
final_price = purchase_amount - discount

# Display results
print(f"Customer Type: {customer_type}")
print(f"Purchase Amount: ${purchase_amount}")
print(f"Discount: ${discount}")
print(f"Final Price: ${final_price}")
