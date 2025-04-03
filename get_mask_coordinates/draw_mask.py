import cv2
import numpy as np

# Global variables so the mouse callback can modify them
drawing_points = []
img_original = None
img_display = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to record points when you left-click on the window.
    Draws small circles for each click, and lines connecting consecutive points.
    """
    global drawing_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))

        # Draw a tiny circle at the clicked point
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)

        # If there's at least one prior point, draw a line to connect them
        if len(drawing_points) > 1:
            cv2.line(img_display, drawing_points[-2], drawing_points[-1], (0, 0, 255), 2)

        cv2.imshow("Select ROI", img_display)

def main():
    global img_original, img_display
    # Load your image
    img_original = cv2.imread("frames/frames121.jpg")  # Change the path to your own image
    if img_original is None:
        print("Could not load the image.")
        return

    # Make a copy for drawing (so we don't alter the original)
    img_display = img_original.copy()

    # Create a window and set the mouse callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    print("Instructions:")
    print("  - Left-click on the image to add polygon points.")
    print("  - Press 'q' or ESC to finish selecting points and create the mask.\n")

    while True:
        cv2.imshow("Select ROI", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC or 'q'
            break

    cv2.destroyWindow("Select ROI")

    # Once you exit, close the polygon if there are enough points
    if len(drawing_points) < 3:
        print("Not enough points to form a polygon!")
        return

    # Optionally draw a line connecting last to first for visual closure
    cv2.line(img_display, drawing_points[-1], drawing_points[0], (0, 0, 255), 2)

    # Create an empty mask
    mask = np.zeros(img_original.shape[:2], dtype=np.uint8)

    # Convert list of tuples to the shape fillPoly expects (N, 1, 2)
    pts_np = np.array(drawing_points, dtype=np.int32).reshape((-1, 1, 2))

    # Fill the polygon on the mask with white
    cv2.fillPoly(mask, [pts_np], 255)

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img_original, img_original, mask=mask)

    # Show final results
    cv2.imshow("Polygon on Image", img_display)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked Image", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the chosen points
    print("Polygon coordinates:")
    for idx, pt in enumerate(drawing_points):
        print(f" Point {idx+1}: (x={pt[0]}, y={pt[1]})")

if __name__ == "__main__":
    main()
