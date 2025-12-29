a = int(input("Enter coefficient of x^2 = "))
b = int(input("Enter coefficient of x = "))
c = int(input("Enter constant term = "))
D = (b*b)-4*(a*c)
print("Value of Discriminant is =", D)
u = ((-b) + ((D)**(0.5)))/(2*a)
U = ((-b) - ((D)**(0.5)))/(2*a)
if D>0:
    print("The first root of the equation is =", u)
    print("The second root of the equation is =", U)
    print("The roots are Real and Unequal")
elif D==0:
    print("The first root of the equation is =", u)
    print("The second root of the equation is =", U)
    print("The roots of the equation are equal")
else:
    print("Roots are Imaginary")
     

