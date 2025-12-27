#Assignment day 22nd OCtober 2025
# No 6: Website log
# Visitor IDs for each day
day1 = {101, 102, 103, 104}
day2 = {103, 104, 105, 106}

# (a) Unique visitors across both days 
unique_visitors = day1.__or__(day2)

# (b) Visitors present on both days
common_visitors = day1.__and__(day2)

# Display the results
print("🧾 Website Visitor Log Analysis")
print("-----------------------------------")
print(f"a) Number of unique visitors across both days: {len(unique_visitors)}")
print(f"   Unique Visitor IDs: {unique_visitors}")
print()
print(f"b) Visitors present on both days: {common_visitors}")
