import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from numpy.linalg import inv

# ... (Your existing code)

def on_pick(event):
    # Check if the picked object is a control point
    if isinstance(event.artist, plt.PathCollection):
        ind = event.ind[0]
        print(f'Picked control point {ind}')

def on_motion(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        # Update the position of the picked control point
        if picked_point is not None:
            origCtrlPoints[picked_point] = [x, y]
            update_curves()

def update_curves():
    # Update Bezier curves based on the new control points
    # (Your existing code to calculate and plot Bezier curves)

    # Redraw the figure
    plt.draw()

# ... (Your existing code)

# Scatter plot for control points (make them pickable)
control_points = ax1.scatter(origCtrlPoints[:, 0], origCtrlPoints[:, 1], color='lightblue', label='Control Points (Original)', s=30, picker=True)

# Event connection for picking control points
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# ... (Your existing code)
