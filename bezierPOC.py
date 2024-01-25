import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.linalg import inv

# Global variables to pass to graphics section later
xOrig, yOrig, xInter, yInter, origCtrlPoints, interpolCtrlPoints = None, None, None, None, None, None

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

def calculateBezierPoints(b, controlPoints):

    bezierPoints = np.zeros((len(b), 2))

    for n in range(len(b)):
        bezierPoints[n, 0] = (1 - b[n])**3 * controlPoints[0, 0] + 3 * (1 - b[n])**2 * b[n] * controlPoints[1, 0] + 3 * (1 - b[n]) * b[n]**2 * controlPoints[2, 0] + b[n]**3 * controlPoints[3, 0]
        bezierPoints[n, 1] = (1 - b[n])**3 * controlPoints[0, 1] + 3 * (1 - b[n])**2 * b[n] * controlPoints[1, 1] + 3 * (1 - b[n]) * b[n]**2 * controlPoints[2, 1] + b[n]**3 * controlPoints[3, 1]

    return bezierPoints

def calculateLeastSquaresSum(originalPoints, bezierPoints):
    
    sum_squared_diff = 0.0

    for n in range(len(originalPoints)):
        # Calculate squared difference for each point and add to the sum
        diff_squared = np.sum((originalPoints[n, :] - bezierPoints[n, :]) ** 2)
        sum_squared_diff += diff_squared

    return sum_squared_diff

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
    print(f'd matrix:\n{d}')

    # calculate the values for the b (Bezier index) matrix, which stores the respective indexes of the points on the cubic Bezier curve
    b = np.zeros(x.size)
    # this loop doesn't iterate over the first value in b, since b[0] = 0
    for i in range(1, x.size):
        # b[i] = length of most recent segment / length of all segments
        b[i] = (d[i]) / d[d.size - 1]
    print(f'b matrix:\n{b}')

    # calculate the values for s (least squares) matrix, which stores the values needed for a least squares regression analysis
    # the last column is filled with ones (b[i]^0)
    s = np.ones((x.size, 4))
    for i in range(0, b.size):
        s[i, 0] = b[i] ** 3
        s[i, 1] = b[i] ** 2
        s[i, 2] = b[i] ** 1
    print(f's matrix:\n{s}')

    # calculate the values for the p array that gives us the x and y locations of the final control points
    # using the following equation, points are calculated: inv(m) * inv(s.T * s) * s.T * (y or x)
    # actual calculations
    xP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), x)
    yP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), y)

    # create matrix p, in which control points will be stored
    controlPoints = np.hstack((xP, yP))

    print(f"\nControl Points:\n{controlPoints}")

    print(f"\nBezier Points: \n{calculateBezierPoints(b, controlPoints)}")

    print(f"\nLeast Squares Sum:\n{calculateLeastSquaresSum(calculateBezierPoints(b, controlPoints), np.hstack((x, y)))}")

    # print(f"\nOriginal Points:\n{np.hstack((x, y))}")

    return controlPoints

def fill_subplot(ax, show_original=True, show_interpolated=True, show_orig_control=True, show_interpol_control=True):
    global xOrig, yOrig, xInter, yInter, origCtrlPoints, interpolCtrlPoints
    
    # Plot Original Points
    if show_original:
        ax.scatter(xOrig, yOrig, color='blue', label='Original Points', s=30)

    # Plot Interpolated Points
    if show_interpolated:
        ax.scatter(xInter, yInter, color='red', label='Interpolated Points', s=6)

    # Plot Bézier Curves
    tValues = np.linspace(0, 1, 100)
    bezierCurveOrig = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
    fitCurveOrig = np.dot(bezierCurveOrig, origCtrlPoints)
    if show_original:
        ax.plot(fitCurveOrig[:, 0], fitCurveOrig[:, 1], color='green', label='Bézier Curve (Original)')

    bezierCurveInter = np.array([[(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3] for t in tValues])
    fitCurveInter = np.dot(bezierCurveInter, interpolCtrlPoints)
    if show_interpolated:
        ax.plot(fitCurveInter[:, 0], fitCurveInter[:, 1], color='purple', label='Bézier Curve (Interpolated)')

    # Plot Control Points
    if show_orig_control:
        ax.scatter(origCtrlPoints[:, 0], origCtrlPoints[:, 1], color='lightblue', label='Control Points (Original)', s=30)

    if show_interpol_control:
        ax.scatter(interpolCtrlPoints[:, 0], interpolCtrlPoints[:, 1], color='lightcoral', label='Control Points (Interpolated)', s=30)

    # Draw thin line segments between adjacent control points
    for i in range(origCtrlPoints.shape[0] - 1):
        if show_orig_control:
            ax.plot([origCtrlPoints[i, 0], origCtrlPoints[i + 1, 0]], [origCtrlPoints[i, 1], origCtrlPoints[i + 1, 1]], color='gray', linestyle='--', linewidth=1)

        if show_interpol_control:
            ax.plot([interpolCtrlPoints[i, 0], interpolCtrlPoints[i + 1, 0]], [interpolCtrlPoints[i, 1], interpolCtrlPoints[i + 1, 1]], color='gray', linestyle='--', linewidth=1)

# Define matrix m, which contains coefficients for the cubic Bezier curve
m = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

# Define the number of points
numPoints = 4

# Define matrix xOrig and yOrig
xOrig = np.linspace(0, 100, numPoints, endpoint=False).reshape(-1, 1)
yOrig = np.random.uniform(0, 100, size=(numPoints, 1))

# With the original points, calculate points of best-fit cubic Bezier curve
origCtrlPoints = calculateLeastSquaresBezier(xOrig, yOrig)

# Call interpolation method
xInter, yInter = interpolatePointsRegularIntervals(xOrig, yOrig, 6)

# With the interpolated points, calculate points of best-fit cubic Bezier curve
interpolCtrlPoints = calculateLeastSquaresBezier(xInter, yInter)

# Create a matplotlib figure with two rows and three columns
fig, axs = plt.subplots(2, 3, figsize=(18, 8.75), sharex='all', sharey='all')

# Assign each subplot to variables for easier reference
ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

# Fill subplots using the new method
fill_subplot(ax1, show_interpolated=False, show_interpol_control=False, show_orig_control=False)
fill_subplot(ax2, show_interpolated=False, show_interpol_control=False)
fill_subplot(ax3)
fill_subplot(ax4, show_orig_control=False, show_interpol_control=False, show_original=False)
fill_subplot(ax5, show_orig_control=False, show_original=False)

# Display the legend in the bottom rightmost subplot
ax6.axis('off')

# Grab the labels from ax1, which has everything turned on
handles, labels = ax3.get_legend_handles_labels()

# use the labels and handels from ax1 to populate the legend appearing in ax6 so that the legend isn't blank
ax6.legend(handles, labels, fontsize='medium', loc='center')

# Tighten the layout
plt.tight_layout()

# Show the figure
plt.show()