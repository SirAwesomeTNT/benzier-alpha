import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.linalg import inv

def interpolatePointsRegularIntervals(xOrig, yOrig, totalPoints):
    # Fill x-axis with regularly spaced values
    xNew = np.linspace(xOrig[0], xOrig[-1], totalPoints)

    # Initialize array for yNew
    yNew = np.zeros_like(xNew)

    # Calculate y-values on line segments connecting the original points
    for i in range(len(xOrig) - 1):
        mask = (xNew >= xOrig[i]) & (xNew <= xOrig[i + 1])
        xSegment = xNew[mask]

        # Linear interpolation for y-values
        yInterp = yOrig[i] + (yOrig[i + 1] - yOrig[i]) / (xOrig[i + 1] - xOrig[i]) * (xSegment - xOrig[i])

        # Update yNew array
        yNew[mask] = yInterp

    # Return the xNew and yNew arrays
    return xNew, yNew

def calculateLeastSquaresBezier(x, y):
    # calculate the values for the d (distance) matrix, which stores the distance from the start of the parent curve to each consecutive point
    d = np.zeros(x.size)
    # this loop doesn't iterate over the first value in d, since d[0] = 0
    for i in range(1, x.size):
        x1 = np.ndarray.item(x[i])
        x2 = np.ndarray.item(x[i - 1])
        y1 = np.ndarray.item(y[i])
        y2 = np.ndarray.item(y[i - 1])
        # a^2 + b^2 = c^2, solving for c
        d[i] = d[i - 1] + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # calculate the values for the b (Bezier index) matrix, which stores the respective indexes of the points on the cubic Bezier curve
    b = np.zeros(x.size)
    # this loop doesn't iterate over the first value in b, since b[0] = 0
    for i in range(1, x.size):
        # b[i] = length of most recent segment / length of all segments
        # broken version (according to Herold's formulas): b[i] = (d[i] - d[i-1]) / d[d.size - 1]
        b[i] = (d[i]) / d[d.size - 1]

    # calculate the values for s (least squares) matrix, which stores the values needed for a least squares regression analysis
    # the last column is filled with ones (b[i]^0)
    s = np.ones((x.size, 4))
    for i in range(0, b.size):
        s[i, 0] = b[i] ** 3
        s[i, 1] = b[i] ** 2
        s[i, 2] = b[i] ** 1

    # calculate the values for the p array that gives us the x and y locations of the final control points
    # using the following equation, points are calculated: inv(m) * inv(s.T * s) * s.T * (y or x)
    # actual calculations
    xP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), x)
    yP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), y)

    # create matrix p, in which control points will be stored
    p = np.hstack((xP, yP))

    return p

# Define matrix m, which contains coefficients for the cubic Bezier curve
m = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

# Define matrix xOrig and yOrig, in which we will store the points of our parent curve (the curve we are modeling)
# yOrig is random, between 0 and 5
xOrig = np.array([[0], [1], [2], [3], [4], [5]])
yOrig = np.random.uniform(1, 3, size=(6, 1))

# With the original 6 points, calculate points of best-fit cubic Bezier curve
pOrig = calculateLeastSquaresBezier(xOrig, yOrig)

# Call interpolation method
xInter, yInter = interpolatePointsRegularIntervals(xOrig, yOrig, 100)

# Calculate points of best-fit cubic Bezier curve
pInter = calculateLeastSquaresBezier(xInter, yInter)

# Create a matplotlib figure with three subplots
plt.figure(figsize=(18, 6))

# Subplot 1: Original and Interpolated Points
plt.subplot(1, 3, 1)

# Plot Original Points
plt.scatter(xOrig, yOrig, color='blue', label='Original Points', s=20)

# Plot Interpolated Points
plt.scatter(xInter, yInter, color='red', label='Interpolated Points', s=8)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Original and Interpolated Points')

# Subplot 2: Bézier Curves
plt.subplot(1, 3, 2)

# Plot Original Points
plt.scatter(xOrig, yOrig, color='blue', label='Original Points', s=20)

# Plot Interpolated Points
plt.scatter(xInter, yInter, color='red', label='Interpolated Points', s=8)

# Plot Bézier Curves
tValues = np.linspace(0, 1, 100)
bezierCurveOrig = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
fitCurveOrig = np.dot(bezierCurveOrig, pOrig)
plt.plot(fitCurveOrig[:, 0], fitCurveOrig[:, 1], color='green', label='Bézier Curve (Original)')

bezierCurveInter = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
fitCurveInter = np.dot(bezierCurveInter, pInter)
plt.plot(fitCurveInter[:, 0], fitCurveInter[:, 1], color='purple', label='Bézier Curve (Interpolated)')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Bézier Curves')

# Subplot 3: Bézier Curves with Control Points
plt.subplot(1, 3, 3)

# Plot Original Points
plt.scatter(xOrig, yOrig, color='blue', label='Original Points', s=20)

# Plot Interpolated Points
plt.scatter(xInter, yInter, color='red', label='Interpolated Points', s=8)

# Plot Bézier Curves
plt.plot(fitCurveOrig[:, 0], fitCurveOrig[:, 1], color='green', label='Bézier Curve (Original)')
plt.plot(fitCurveInter[:, 0], fitCurveInter[:, 1], color='purple', label='Bézier Curve (Interpolated)')

print(pOrig)

# Plot Control Points
plt.scatter(pOrig[:, 0], pOrig[:, 1], color='lightblue', label='Control Points (Original)', s=30)
plt.scatter(pInter[:, 0], pInter[:, 1], color='lightcoral', label='Control Points (Interpolated)', s=30)

# Draw thin line segments between adjacent control points
for i in range(pOrig.shape[0] - 1):
    plt.plot([pOrig[i, 0], pOrig[i + 1, 0]], [pOrig[i, 1], pOrig[i + 1, 1]], color='gray', linestyle='--', linewidth=1)
    plt.plot([pInter[i, 0], pInter[i + 1, 0]], [pInter[i, 1], pInter[i + 1, 1]], color='gray', linestyle='--', linewidth=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Bézier Curves with Control Points')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the figure
plt.show()