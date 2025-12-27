#Assignment day 22nd OCtober 2025
# No 4: Teacher-student list

# List of student grades
grades = [88, 92, 76, 95, 61, 88]

print("Original Grades:", grades)

# Remove the lowest grade
lowest_grade = min(grades)
grades.remove(lowest_grade)

print("Lowest grade removed:", lowest_grade)
print("Updated Grades:", grades)

# Average of remaining grades
average = sum(grades) / len(grades)

# Rounding to one d.p.
average = round(average, 1)

print(f"The average of the remaining grades is {average}")
