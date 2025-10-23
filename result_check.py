score = 80
attendance = 75
submitted = True
if score >= 60:
    if attendance >=65:
        if submitted:
            print("Pass with good grades")
        else:
            print("Pass but missing assignment")
    else:
            print("Pass but low attendance")
else:
            print("failed")