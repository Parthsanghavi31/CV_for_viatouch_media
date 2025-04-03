import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import percentileofscore
from copy import deepcopy
import sys
def interactive_frames_and_difference_plot(frames, filtered_images, subtracted_images, norms, output_file, save_video= True):
    """
    Displays an interactive window that shows:
      - The *original* frame (left)
      - The *difference* image (right)
      - A line plot of norms (bottom)
      - A slider to scroll through frames manually
      - Play and Pause buttons to animate from the current slider position to the end (no looping).
    
    Parameters
    ----------
    frames : list of original frames (BGR) from get_frames_from_video
    subtracted_images : list of grayscale difference images from process_frames
    norms : list or array of norm values for each difference image
    """

    if frames == None or subtracted_images == None or filtered_images == None:
        print("No frames, subtracted images, or norms found.")
        return

    # Convert original frames (for the relevant indices) to RGB
    # Because subtracted_images go from index=1..(len(frames)-1),
    # we'll map each difference image "i" to frames[i+1] or frames[i].
    # For consistency with how process_frames is enumerated, let's use frames[1:-1].
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames[10:-1]]

    # Convert the grayscale difference images to RGB for Matplotlib display
    difference_rgb = [img for img in subtracted_images]
    
    filtered_img = [img for img in filtered_images]
    
    # Convert norms to a numpy array (if not already)
    norm_array = np.array(norms)
    frame_indices = np.arange(len(norm_array))

    # --- Create the figure and axes layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[8,5,5,1])

    ax_orig = fig.add_subplot(gs[0, 0])      # top-left
    ax_diff = fig.add_subplot(gs[0, 1])      # top-right
    ax_filtered = fig.add_subplot(gs[1,0]) # second row-left
    ax_plot_orignal = fig.add_subplot(gs[2, :])      # entire bottom row
    
    # Display the first original frame and the first difference image
    current_orig_im = ax_orig.imshow(original_rgb[0])
    ax_orig.set_title("Original Frame")
    ax_orig.axis("off")

    current_diff_im = ax_diff.imshow(difference_rgb[0])
    ax_diff.set_title("Difference Image (Frame n vs. Frame n+1)")
    ax_diff.axis("off")
    
    filltered_im = ax_filtered.imshow(filtered_img[0])
    ax_filtered.set_title("Filtered Frame")
    ax_filtered.axis("off")

    # Plot the norms on the bottom subplot
    ax_plot_orignal.plot(frame_indices, norm_array, label="Frame Difference Norm", color='b')
    ax_plot_orignal.set_title("Frame-to-Frame Grayscale Difference")
    ax_plot_orignal.set_xlabel("Subtracted Image Index")
    ax_plot_orignal.set_ylabel("Norm of Difference")
    ax_plot_orignal.grid(True)
    ax_plot_orignal.legend()

    # A vertical line that we'll move to indicate the current index
    marker_line = ax_plot_orignal.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # --- Slider to navigate frames manually ---
    # Place the slider below the bottom row
    slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=len(subtracted_images) - 1,
        valinit=0,
        valstep=1,
        color='lightblue'
    )
    
    # --- Saving the Video ---
    if save_video:
        frame_size = (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi
        out = cv2.VideoWriter(output_file, fourcc, 20.0, frame_size)
    
    # --- Closing the plot usng ESC key --- 
    def on_keypress(event):
        """Closes the Matplotlib window when 'Esc' key is pressed."""
        if event.key == 'escape':
            plt.close(fig)

    # Connect the keypress event to the figure
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    
    # --- Buttons for Play and Pause ---
    play_pause_ax = plt.axes([0.45, 0.01, 0.15, 0.05])
    play_pause_button = Button(play_pause_ax, 'Play', color='lightgreen', hovercolor='green')

    # Use lists so nested functions can modify these variables
    is_playing = [False]
    current_index = [0]

    def update_slider(idx):
        """Sets the slider to a new index and triggers the display update."""
        slider.set_val(idx)  # will call slider_update
        fig.canvas.draw_idle()

    def slider_update(val):
        """Callback for manual slider changes or forced updates via set_val."""
        idx = int(slider.val)
        # Update both images
        current_orig_im.set_data(original_rgb[idx])
        current_diff_im.set_data(difference_rgb[idx])
        filltered_im.set_data(filtered_img[idx])
        # Update the vertical line in the plot
        marker_line.set_xdata([idx, idx])
        # Store current index
        current_index[0] = idx

    slider.on_changed(slider_update)

    def play_pause_event(event):
        """Toggles between playing and pausing."""
        if is_playing[0]:  # If currently playing, pause it
            is_playing[0] = False
            play_pause_button.label.set_text('Play')
        else:  # If paused, start playing
            is_playing[0] = True
            play_pause_button.label.set_text('Pause')
            start_idx = current_index[0]
            for idx in range(start_idx, len(subtracted_images)):
                if not is_playing[0]:  # Stop if user pauses
                    break
                update_slider(idx)
                plt.pause(0.03)  # Adjust speed as needed
                
                # --- Capture and Save Frame (if recording is enabled) ---
                if save_video:
                    fig.canvas.draw()
                    frame = np.array(fig.canvas.buffer_rgba())
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    out.write(frame)

            is_playing[0] = False  # Reset state when done
            play_pause_button.label.set_text('Play')

    play_pause_button.on_clicked(play_pause_event)

    # Initialize at index=0
    slider_update(0)
    plt.show()
    
    # Release video writer if enabled
    if save_video:
        out.release()
        print(f"Video saved as {output_file}")
  
#------------------------------------------------------------------------------------------------------------------------# 

def crop_frames(frames, cropping, cropped_frames = []):

    dimension = frames[0].shape
    for i in range(1,len(frames)-1):
        cropped_frame = frames[i][0:140, 0:dimension[1]] # for Data_MD videos
        
        # cropped_frame = frames[i][0:150, 0:dimension[1]] # for Queens_bev videos
        # cropped_frame = frames[i][36:dimension[0], :] # for Test_vid1 videos

        cropped_frames.append(cropped_frame)
    if cropping == True:
        return cropped_frames
    
    return frames

def get_frames_from_video(video_path, frames = None):
    
    if frames is None:
        frames = []
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count<=3000:
        ret, frame = video.read()
        if not ret:
            break        
        frames.append(frame)
        frame_count+=1
        """path = f"frames/frames{frame_count}.jpg"
        cv2.imwrite(path, frame)"""
        
    video.release()
    cv2.destroyAllWindows()
    
    return frames

def process_frames(frames, norm = [], subtracted_images= [], filtered_images_spatial = []):
    
    for idx in range(10, len(frames)-10):
        
        current_frame  = frames[idx]
        next_frame = frames[idx+1]
        
        # Converting images to Grayscale and normalizing them
        gs_current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        gs_current_frame_normalized = gs_current_frame
        gs_next_frame = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
        gs_next_frame_normalized = gs_next_frame
        
        # Applying Gaussian blur to remove noise.
        kernel = np.ones((5,5),np.float32)/25
        gaussian_blur_curr = cv2.filter2D(gs_current_frame_normalized,-1,kernel)
        gaussian_blur_next = cv2.filter2D(gs_next_frame_normalized, -1, kernel)
        
        # Taking difference of consecutive images to detect motion
        subtracted_image = cv2.absdiff(gaussian_blur_curr, gaussian_blur_next)
        subtracted_images.append(subtracted_image)
        
        filtered_image_spatial = spatial_domain_filtering(subtracted_image)
        filtered_images_spatial.append(filtered_image_spatial)
        # Finding norm of the difference
        norm.append(np.linalg.norm(subtracted_image))
        
    return filtered_images_spatial, subtracted_images, norm

def spatial_domain_filtering(subtracted_image):
    # Applying threshold to get a binary image
    
    # thres = cv2.threshold(subtracted_image, 100)
    filtered_image = subtracted_image
    # filtered_image = cv2.bilateralFilter(subtracted_image, 10, 50, 50)
    # thres = cv2.threshold(normalized_image, 25, 500, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0.5)
    median_filtered_img = cv2.medianBlur(cv2.bitwise_not(threshold), 3)
    
    # Applying morphological operations
    elipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # dilated_image = cv2.dilate(median_filtered_img, elipse_kernel, iterations=1)
    morphed_image =  cv2.morphologyEx(median_filtered_img, cv2.MORPH_CLOSE, elipse_kernel, iterations=2)
    contours, hierarchy = cv2.findContours(
        morphed_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    black_canvas = np.zeros((morphed_image.shape[0], morphed_image.shape[1], 3), dtype=np.uint8)

    # Set a minimum contour area to filter smaller contours (if required)
    min_area = 2000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # Draw contours onto your black canvas
    cv2.drawContours(black_canvas, filtered_contours, -1, (0, 255, 0), 3)

    return black_canvas

def frequency_domain_filtering():
    pass
    
def main(video_name):
    
    video_path = f"data/{video_name}"
    original_frames = get_frames_from_video(video_path)
    cropped_frames = crop_frames(original_frames, cropping=True)
    filtered_images, subtracted_images, norms = process_frames(cropped_frames)
    
    interactive_frames_and_difference_plot(cropped_frames, filtered_images, subtracted_images, norms, output_file=f"processed_videos/{video_name}", save_video= False)
   

if __name__ == '__main__':
    video_name = "Queens_beverage_3.mp4"
    main(video_name)

