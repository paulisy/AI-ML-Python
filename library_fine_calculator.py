#Assignment day 22nd OCtober 2025
# No 10: Nested if statement and pass statement

def  calculate_library_fine(days_late):
    if days_late <= 0:
        fine = 0
    else:
        if days_late <= 7:
            fine = days_late * 0.50
        else:
            fine = 7 * 0.50  #fine
#for the first 7 days
        if days_late <= 30:
            fine += (days_late - 7) * 1.00  #addtional fine for days 9 t0 30
        else:
            pass    #Account suspended, no fine calculated
            fine = None #Set fine to None for Clarity
    return fine

##############################################################
days_late = int(input("Please enter the number of days the book is late:"))
fine = calculate_library_fine(days_late)
if fine is None:
      print("Account suspended. No fine calculated.")
else:
    print(f"The total fine is: ${fine:.2f}")