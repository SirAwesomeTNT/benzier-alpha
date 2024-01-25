import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.linalg import inv

# Global variables to pass to the graphics section later
xOrig, yOrig, xInter, yInter, origCtrlPoints, interCtrlPoints = None, None, None, None, None, None

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

def calculatePointsOnBezier(bezierIndex, controlPoints):
    # Calculate points on the Bézier curve based on given bezierIndex matrix and control points
    bezierPoints = np.zeros((len(bezierIndex), 2))

    for n in range(len(bezierIndex)):
        # x-values
        bezierPoints[n, 0] = (1 - bezierIndex[n])**3 * controlPoints[0, 0] + 3 * (1 - bezierIndex[n])**2 * bezierIndex[n] * controlPoints[1, 0] + 3 * (1 - bezierIndex[n]) * bezierIndex[n]**2 * controlPoints[2, 0] + bezierIndex[n]**3 * controlPoints[3, 0]
        # y-values
        bezierPoints[n, 1] = (1 - bezierIndex[n])**3 * controlPoints[0, 1] + 3 * (1 - bezierIndex[n])**2 * bezierIndex[n] * controlPoints[1, 1] + 3 * (1 - bezierIndex[n]) * bezierIndex[n]**2 * controlPoints[2, 1] + bezierIndex[n]**3 * controlPoints[3, 1]

    return bezierPoints

def calculateLeastSquaresSum(originalPoints, bezierPoints):
    # Calculate the sum of squared differences between original points and corresponding Bézier curve points
    sum_squared_diff = 0.0

    for n in range(len(originalPoints)):
        # Calculate squared difference for each point and add to the sum
        diff_squared = np.sum((originalPoints[n, :] - bezierPoints[n, :]) ** 2)
        sum_squared_diff += diff_squared

    return sum_squared_diff

def calculateDistanceMatrix(x, y):
    # Calculate the values for the distance matrix, which stores the distance from the start of the parent curve to each consecutive point
    distance = np.zeros(x.size)

    # This loop doesn't iterate over the first value in d since d[0] = 0
    for i in range(1, x.size):
        x1 = np.ndarray.item(x[i])
        x2 = np.ndarray.item(x[i - 1])
        y1 = np.ndarray.item(y[i])
        y2 = np.ndarray.item(y[i - 1])

        # a^2 + b^2 = c^2, solving for c
        distance[i] = distance[i - 1] + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    return distance

def calculateBezierIndexMatrix(x, y):
    # Calculate the values for the bezierIndex matrix, which stores the respective indexes of the points on the cubic Bezier curve

    # Calculate distance matrix
    distance = calculateDistanceMatrix(x, y)

    # Initialize matrix
    bezierIndex = np.zeros(x.size)

    # This loop doesn't iterate over the first value in bezierIndex, since bezierIndex[0] = 0
    for i in range(1, x.size):
        # bezierIndex[i] = length of the most recent segment / length of all segments
        bezierIndex[i] = (distance[i]) / distance[distance.size - 1]
    
    return bezierIndex

def calculateLeastSquaresCoefficientsMatrix(x, y):
    # Calculate the values for leastSquaresCoefficients matrix, which stores the values needed for a least squares regression analysis
    # The last column is filled with ones (bezierIndex[i]^0)

    bezierIndex = calculateBezierIndexMatrix(x, y)

    leastSquaresCoefficients = np.ones((x.size, 4))
    for i in range(0, bezierIndex.size):
        leastSquaresCoefficients[i, 0] = bezierIndex[i] ** 3
        leastSquaresCoefficients[i, 1] = bezierIndex[i] ** 2
        leastSquaresCoefficients[i, 2] = bezierIndex[i] ** 1

    return leastSquaresCoefficients

def calculateLeastSquaresBezierControlPoints(x, y):

    s = calculateLeastSquaresCoefficientsMatrix(x, y)

    # Calculate the values for the p array that gives us the x and y locations of the final control points
    # Using the following equation, points are calculated: inv(m) * inv(s.T * s) * s.T * (y or x)
    # Actual calculations
    xP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), x)
    yP = np.matmul(np.matmul(np.matmul(inv(m), inv(np.matmul(s.T, s))), s.T), y)

    # Create matrix p, in which control points will be stored
    controlPoints = np.hstack((xP, yP))

    return controlPoints

def fillSubplot(ax, show_original=True, show_interpolated=True, show_orig_control=True, show_interpol_control=True):
    global xOrig, yOrig, xInter, yInter, origCtrlPoints, interCtrlPoints
    
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
    fitCurveInter = np.dot(bezierCurveInter, interCtrlPoints)
    if show_interpolated:
        ax.plot(fitCurveInter[:, 0], fitCurveInter[:, 1], color='purple', label='Bézier Curve (Interpolated)')

    # Plot Control Points
    if show_orig_control:
        ax.scatter(origCtrlPoints[:, 0], origCtrlPoints[:, 1], color='lightblue', label='Control Points (Original)', s=30)

    if show_interpol_control:
        ax.scatter(interCtrlPoints[:, 0], interCtrlPoints[:, 1], color='lightcoral', label='Control Points (Interpolated)', s=30)

    # Draw thin line segments between adjacent control points
    for i in range(origCtrlPoints.shape[0] - 1):
        if show_orig_control:
            ax.plot([origCtrlPoints[i, 0], origCtrlPoints[i + 1, 0]], [origCtrlPoints[i, 1], origCtrlPoints[i + 1, 1]], color='gray', linestyle='--', linewidth=1)

        if show_interpol_control:
            ax.plot([interCtrlPoints[i, 0], interCtrlPoints[i + 1, 0]], [interCtrlPoints[i, 1], interCtrlPoints[i + 1, 1]], color='gray', linestyle='--', linewidth=1)

def showGraphics():
    # Create a matplotlib figure with two rows and three columns
    # fig is a matplotlib figure, it is very necessary even though it isn't accessed
    fig, axs = plt.subplots(2, 3, figsize=(18, 8.75), sharex='all', sharey='all')

    # Assign each subplot to variables for easier reference
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

    # Fill subplots using the new method
    fillSubplot(ax1, show_interpolated=False, show_interpol_control=False, show_orig_control=False)
    fillSubplot(ax2, show_interpolated=False, show_interpol_control=False)
    fillSubplot(ax3)
    fillSubplot(ax4, show_orig_control=False, show_interpol_control=False, show_original=False)
    fillSubplot(ax5, show_orig_control=False, show_original=False)

    # Display the legend in the bottom rightmost subplot
    ax6.axis('off')

    # Grab the labels from ax3, which has everything turned on
    handles, labels = ax3.get_legend_handles_labels()

    # Use the labels and handles from ax3 to populate the legend appearing in ax6 so that the legend isn't blank
    ax6.legend(handles, labels, fontsize='medium', loc='center')

    # Calculate points on each Bezier curve that corrolate with initial points
    origPointsOnBezier = calculatePointsOnBezier(calculateBezierIndexMatrix(xOrig, yOrig), origCtrlPoints)
    # Calculate a least squares sum for the difference between the two
    origLeastSquaresSum = calculateLeastSquaresSum(origPointsOnBezier, np.hstack((xOrig, yOrig)))

    # Do the same for the interpolated points and curve
    interPointsOnBezier = calculatePointsOnBezier(calculateBezierIndexMatrix(xInter, yInter), interCtrlPoints)
    interLeastSquaresSum = calculateLeastSquaresSum(interPointsOnBezier, np.hstack((xInter, yInter)))

    # Add a display box below the legend with the values of the interpolated and original least square sum
    display_box_text = f'Original Least Squares Sum: {origLeastSquaresSum:.4f}' \
                    f'\nInterpolated Least Squares Sum: {interLeastSquaresSum:.4f}' \
                    f'\nOriginal Least Squares Sum per Point: {origLeastSquaresSum / origNumPoints:.4f}' \
                    f'\nInterpolated Least Squares Sum per Point: {interLeastSquaresSum / interNumPoints:.4f}'
    ax6.text(0.06, 0.25, display_box_text, transform=ax6.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tighten the layout
    plt.tight_layout()

    # Show the figure
    plt.show()

# Define matrix m, which contains coefficients for the cubic Bézier curve
m = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

# Define the number of points
origNumPoints = 4
interNumPoints = 10

# Define matrix xOrig and yOrig (randomly generate original points)
xOrig = np.linspace(0, 100, origNumPoints, endpoint=False).reshape(-1, 1)
yOrig = np.random.uniform(0, 100, size=(origNumPoints, 1))

# With the original points, calculate the control points of the best-fit cubic Bézier curve using Least Squares
origCtrlPoints = calculateLeastSquaresBezierControlPoints(xOrig, yOrig)

# Interpolate a new set of points based on the original points
xInter, yInter = interpolatePointsRegularIntervals(xOrig, yOrig, interNumPoints)

# With the interpolated points, calculate points of the best-fit cubic Bézier curve
interCtrlPoints = calculateLeastSquaresBezierControlPoints(xInter, yInter)

# Show graphics
showGraphics()