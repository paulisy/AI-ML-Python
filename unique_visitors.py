#Assignment day 22nd OCtober 2025
# No 5: Unique visitors
# Lists of visitors
visitors_today = ["Alice", "Bob", "Charlie", "Alice"]
visitors_yesterday = ["Bob", "David", "Eve"]

# To remove any duplicates convert lists to sets
set_today = set(visitors_today)
set_yesterday = set(visitors_yesterday)

# Find all unique visitors (Union of both sets) (union is same as or)
unique_visitors = set_today.union(set_yesterday)

# Display the result
print("🧾 Unique Website Visitors")
print("----------------------------")
print(f"Visitors today: {set_today}")
print(f"Visitors yesterday: {set_yesterday}")
print(f"\nUnique visitors (today or yesterday): {unique_visitors}")
