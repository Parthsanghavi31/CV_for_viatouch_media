import cv2
import numpy as np
import os
import json
import matplotlib
from datetime import datetime
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


def interactive_frames_and_difference_plot(frames, mog_images): #, frame_diff, canny_edges, subtracted_canny_images, sub_norms, canny_norms, digital_signal, load_sensor_events):
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
    frame_diff : list of grayscale difference images from process_frames
    norms : list or array of norm values for each difference image
    """

    # if frames == None or frame_diff == None or canny_edges == None or canny_norms == None:
    #     print("No frames, subtracted images, or frame_diff found.")
    #     return

    # Convert original frames (for the relevant indices) to RGB
    # Because frame_diff go from index=1..(len(frames)-1),
    # we'll map each difference image "i" to frames[i+1] or frames[i].
    # For consistency with how process_frames is enumerated, let's use frames[1:-1].
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Convert the grayscale difference images to RGB for Matplotlib display
    mog_img = [img for img in mog_images]

    #Digital Data
    # --- Create the figure and axes layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(1, 2, height_ratios=[12])

    ax_orig = fig.add_subplot(gs[0, 0])      # top-left
    ax_diff = fig.add_subplot(gs[0, 1])      # top-right
   

    # Display the first original frame and the first difference image
    current_orig_im = ax_orig.imshow(original_rgb[0])
    ax_orig.set_title("Original Frame")
    ax_orig.axis("off")

    mog_im = ax_diff.imshow(mog_img[0], cmap='gray', vmin=0, vmax=255)
    ax_diff.set_title("Difference Image (Frame n vs. Frame n+1)")
    ax_diff.axis("off")

    # --- Slider to navigate frames manually ---
    # Place the slider below the bottom row
    slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=len(mog_images)-1,
        valinit=0,
        valstep=1,
        color='lightblue'
    )
    
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
        mog_im.set_data(mog_img[idx])
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
            for idx in range(start_idx, len(mog_images)):
                if not is_playing[0]:  # Stop if user pauses
                    break
                update_slider(idx)
                plt.pause(0.03)  # Adjust speed as needed

            is_playing[0] = False  # Reset state when done
            play_pause_button.label.set_text('Play')

    play_pause_button.on_clicked(play_pause_event)

    # Initialize at index=0
    slider_update(0)
    plt.show()
  
#------------------------------------------------------------------------------------------------------------------------# 


def crop_frames(frames, cropped_frames, cropping):

    # dimension = frames[0].shape
    for i in range(0,len(frames)-1):
        cropped_frame = frames[i][70:120, :] 
        cropped_frames.append(cropped_frame)
    if cropping == True:
        return cropped_frames
    return frames

def get_frames_from_video(video_path, frames = None):
    
    if frames is None:
        frames = []
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    # mog_images = []
    # backsub = cv2.createBackgroundSubtractorKNN(history=50)
    while True:
        ret, frame = video.read()
        if not ret:
            break    
        # if frame_count>=10:   
        frame = cv2.resize(frame, (320,240)) 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(np.average(gray))
        if np.average(gray) < 45:
            continue
        else:
            frames.append(frame)
            # fgmask = backsub.apply(frame)
            # print(fgmask)
            # _, binary_mask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            # print(binary_mask)


            # mog_images.append(binary_mask)
            # 
            # cv2.imshow('mog_img', binary_mask)
            # cv2.waitKey(1)

        frame_count+=1
        # path = f"frames/frames{frame_count}.jpg"
        # cv2.imwrite(path, frame)
        
    video.release()
    cv2.destroyAllWindows()
    return frames

def main():

    transaction = "data_for_Mapping_logic/office_transaction_6"
    video_path = os.path.join(transaction, "media.mp4")
    cropped_frames = []
    cropped_mog_images = []
    
    frames = get_frames_from_video(video_path)
    cropped_frames = crop_frames(frames, cropped_frames, cropping=True)
    start_idx = 10
    end_idx   = len(frames) - 10
    if end_idx <= start_idx:
        print("Not enough frames after cropping; adjust start/end indices.")
        return
    mog_images = apply_mog(frames)
    valid_frames = cropped_frames[start_idx:end_idx]
    
        
    interactive_frames_and_difference_plot(valid_frames, valid_mog_images)
    
if __name__ == '__main__':
    main()