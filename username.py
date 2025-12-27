username = False
password = True
is_active = True

if username:
    if password:
        if is_active:
            print("Login successful")
        else:
            print("Account is not active")
    else:
        print("Password incorrect")
else:
    print("Username incorrect")

u = 16
v = 12
if u < v:
    pass
else:
    print("can't work")
