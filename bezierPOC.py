import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.linalg import inv

# define matrix m, which contains coefficients for the cubic Bezier curve
m = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
print("Matrix m, which which contains coefficients for the cubic Bezier curve:\n" + f"{m}\n")

# define matrix x and y, in which we will store the points of our parent curve (the curve we are modeling)
x = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([[4], [1], [0], [1], [4], [9]])
print("Matrix x, which stores the x-values of our parent curve:\n" + f"{x}\n")
print("Matrix y, which stores the y-values of our parent curve:\n" + f"{y}\n")

# calculate the values for the d (distance) matrix, which stores the distance from the start of the parent curve to each consecutive point
d = np.zeros(x.size)
# this loop doesn't iterate over the first value in d, since d[0] = 0
for i in range (1, x.size):
    x1 = np.ndarray.item(x[i])
    x2 = np.ndarray.item(x[i-1])
    y1 = np.ndarray.item(y[i])
    y2 = np.ndarray.item(y[i-1])
    # a^2 + b^2 = c^2, solving for c
    d[i] = d[i-1] + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
print("Matrix d, which stores the distance from the start of the parent curve to each consecutive point:\n" + f"{d}\n")

# calculate the values for the b (Bezier index) matrix, which stores the respective indexes of the points on the cubic Bezier curve
b = np.zeros(x.size)
# this loop doesn't iterate over the first value in b, since b[0] = 0
for i in range (1, x.size):
    # b[i] = length of most recent segment / length of all segments
    # broken version (according to Herold's formulas): b[i] = (d[i] - d[i-1]) / d[d.size - 1]
    b[i] = (d[i]) / d[d.size - 1]
print("Matrix b, which stores the percentages of each point's distance along on the Bezier curve.")
print("These are also the curve's t values that most closely corrospond to each original point:\n" + f"{b}\n")

# calculate the values for s (least squares) matrix, which stores the values needed for a least squares regression analysis
# the last column is filled with ones (b[i]^0)
s = np.ones((x.size, 4))
for i in range(0, b.size):
    s[i, 0] = b[i] ** 3
    s[i, 1] = b[i] ** 2
    s[i, 2] = b[i] ** 1
print("Matrix s, which stores the values needed for a least squares regression analysis:\n" + f"{s}\n")

# calculate the values for the p array that gives us the x and y locations of the final control points
# using the following equation, points are calculated: inv(m) * inv(s.T * s) * s.T * (y or x)
# actual calculations
xP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), x)
yP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), y)

# create matrix p, in which control points will be stored
p = np.hstack((xP, yP))
pRounded = np.round(p, 2)
print("Matrix p, which stores final control points for the Bezier curve:\n" + f"{pRounded}")

# Create a figure with two subplots (1 row, 2 columns)
plt.figure(figsize=(12, 5))  # Adjust the figure size if needed

# Subplot 1: Original Bézier curve without control points
plt.subplot(1, 2, 1)

# Plot the original points
plt.scatter(x, y, color='blue', label='Original Points')

# Plot the fitted Bézier curve
t_values = np.linspace(0, 1, 100)
bezier_curve = np.array([[(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3] for t in t_values])
fit_curve = np.dot(bezier_curve, p)

plt.plot(fit_curve[:, 0], fit_curve[:, 1], color='red', label='Fitted Bézier Curve')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Original Bézier Curve without Control Points')

# Subplot 2: Bézier curve with control points
plt.subplot(1, 2, 2)

# Plot the original points
plt.scatter(x, y, color='blue', label='Original Points')

# Plot the fitted Bézier curve
plt.plot(fit_curve[:, 0], fit_curve[:, 1], color='red', label='Fitted Bézier Curve')

# Plot the control points
plt.scatter(pRounded[:, 0], pRounded[:, 1], color='green', label='Control Points')

# Draw thin line segments between adjacent control points
for i in range(pRounded.shape[0] - 1):
    plt.plot([pRounded[i, 0], pRounded[i + 1, 0]], [pRounded[i, 1], pRounded[i + 1, 1]], color='gray', linestyle='--')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Bézier Curve with Control Points')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the figure
plt.show()