import cv2
import numpy as np
import math

# Suppose we have a list of centroids for each cluster
# In real usage, you would get these from cv2.findContours or other steps
cluster_centers = [
    (50, 60),
    (120, 80),
    (125, 85),
    (200, 200),
    (205, 205),
    (300, 100),
    (300, 200),
    (300, 250)
]

# Create a blank canvas for visualization
height, width = 400, 400
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the cluster centers on the canvas
for (cx, cy) in cluster_centers:
    cv2.circle(canvas, (cx, cy), 5, (0, 255, 0), -1)

# Distance threshold to decide if two clusters are "close"
DIST_THRESHOLD = 60

# Check pairwise distances, and connect if below threshold
for i in range(len(cluster_centers)):
    for j in range(i + 1, len(cluster_centers)):
        (x1, y1) = cluster_centers[i]
        (x2, y2) = cluster_centers[j]

        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist < DIST_THRESHOLD:
            # Draw a line connecting the centroids
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Show the final result
cv2.imshow("Connected Clusters", canvas)
cv2.waitKey(1000)
cv2.destroyAllWindows()
