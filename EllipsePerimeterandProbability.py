import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from scipy.special import ellipe
from scipy.special import ellipk
from scipy.integrate import quad

def ellipse_perimeter(a, b):
    a_arr = np.maximum(a, b)
    b_arr = np.minimum(a, b)
    e = np.sqrt(1 - (b_arr/a_arr)**2)
    return 4 * a_arr * ellipe(e**2)


#Problem 1
#Part A
a = 5
A0 = np.arange(1, 4.1, 0.1)


#part B

A1 = ellipse_perimeter(5, A0)

#part C
h = 0.1
A2 = (ellipse_perimeter(5, A0 + h) - ellipse_perimeter(5, A0 - h)) / (2 * h)


#part D
max = np.argmax(A2)
A3 = A0[max]
A4 = A2[max]

#Part E
b_target = 3.0
i = np.argmin(np.abs(A0 - b_target))
A5 = A2[i] * 0.1


#part F
b2 = np.linspace(4, 5, 20)
P2 = ellipse_perimeter(5, b2)

A6 = np.empty_like(b2)
A6[:-1] = (P2[1:] - P2[:-1]) / (b2[1] - b2[0])
A6[-1] = A6[-2]


#Part G

A7 = np.logspace(-6, -2, 20)

#Part H
A8 = []
for h in A7:
    P_plus = ellipse_perimeter(5, 3 + h)
    P_minus = ellipse_perimeter(5, 3 - h)
    derivative_approx = (P_plus - P_minus) / (2 * h)
    A8.append(derivative_approx)
A8 = np.array(A8)

#Part I
a = 5
b = 3
e2 = (1 - (b/a)**2)
exact_derivative = 4 * b * (ellipk(e2) - ellipe(e2)) / (a*e2)
A9 = np.abs(np.array(A8) - exact_derivative)

#Problem 2
#Part A
mu = 85
sigma = 8.3

def pdf(x):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Part A: true value using quad
A10, _ = quad(pdf, 110, 130)

# Part B: left-sided rectangle rule
spacing = 2.0 ** (-np.arange(1, 13))
A11 = []
for dx in spacing:
    x = np.arange(110, 130, dx)
    A11.append(np.sum(pdf(x)) * dx)
A11 = np.array(A11)

# Part C: right-sided rectangle rule
A12 = []
for dx in spacing:
    x = np.arange(110 + dx, 130 + dx, dx)
    A12.append(np.sum(pdf(x)) * dx)
A12 = np.array(A12)

# Part D: Simpson’s rule
A13 = []
for k in range(1, 13):
    dx = 2**(-k)
    n = int((130 - 110) / dx)
    if n % 2 == 1:
        n += 1
    x = np.linspace(110, 130, n + 1)
    y = pdf(x)
    S = (dx / 3) * (
        y[0]
        + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-2:2])
    )
    A13.append(S)
A13 = np.array(A13)



#------------------Figure Generation-----------------------#


fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
Left_hand_error = np.abs(A11 - A10)
Right_hand_error = np.abs(A12 - A10)
Simpson_error = np.abs(A13 - A10)

stepsize = np.array([2**(-1), 2**(-2), 2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8), 2**(-9), 2**(-10), 2**(-11), 2**(-12)])

h0 = stepsize[0]
C1 = Left_hand_error[0] / h0
Oh = C1 * stepsize

C4 = Simpson_error[0] / h0**4
Oh4 = C4 * stepsize**4

machine_precision = np.full_like(stepsize, 10e-16)

ax.loglog(stepsize, Left_hand_error, marker='+', linestyle = 'none', label='Left-hand rule Error', color='blue')
ax.loglog(stepsize, Right_hand_error, marker='.', linestyle = 'none', label='Right-hand rule Error', color='green')
ax.loglog(stepsize, Simpson_error, marker='o', linestyle = 'none', label='Simpson rule Error', color='red')
ax.loglog(stepsize, machine_precision, label='Machine Precision', color='black')
ax.loglog(stepsize, Oh, label='Linear Error', linestyle='-', color='purple')
ax.loglog(stepsize, Oh4, label='Quartic Error', linestyle='--', color='orange')
ax.legend()
ax.set_xlabel("Step Size for Numerical Integration")
ax.set_ylabel("Error between Numerical Integration and True Value")
ax.set_title("Error of Numerical Integration Methods vs Step Size")
plt.grid(True)
plt.show()
