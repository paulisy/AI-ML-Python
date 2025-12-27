#Assignment day 22nd OCtober 2025
# No 9: If Statements – Discount Eligibility with Multiple Conditions
# Define customer type and purchase amount
# customer type can be "VIP", "Regular", or "New"
# # Ask user for input
customer_type = input("Enter customer type (VIP / Regular / New): ").strip().capitalize()
purchase_amount = float(input("Enter purchase amount ($): "))

# Initialize discount
discount = 0

# Apply discount rules based on customer type
if customer_type == "Vip":
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
    print("Invalid customer type entered. Please enter VIP, Regular, or New.")

# Calculate and display results only if customer type is valid
if customer_type in ["Vip", "Regular", "New"]:
    final_price = purchase_amount - discount
    print("\n--- Purchase Summary ---")
    print(f"Customer Type: {customer_type}")
    print(f"Purchase Amount: ${purchase_amount:.2f}")
    print(f"Discount Applied: ${discount:.2f}")
    print(f"Final Price to Pay: ${final_price:.2f}")
