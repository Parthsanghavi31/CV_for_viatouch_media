import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.preprocessing import MinMaxScaler



def interactive_frames_and_difference_plot(frames, frame_diff, filtered_images, freq_to_spatial_frames ):
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

    if frames == None or frame_diff == None or filtered_images == None:
        print("No frames, subtracted images, or frame_diff found.")
        return

    # Convert original frames (for the relevant indices) to RGB
    # Because frame_diff go from index=1..(len(frames)-1),
    # we'll map each difference image "i" to frames[i+1] or frames[i].
    # For consistency with how process_frames is enumerated, let's use frames[1:-1].
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Convert the grayscale difference images to RGB for Matplotlib display
    difference_rgb = [img for img in frame_diff]
    
    frequency_domain = [img for img in filtered_images]
    spatial_domain = [img for img in freq_to_spatial_frames]
   
    
    #Digital Data
    # --- Create the figure and axes layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[12,12,1])

    ax_orig = fig.add_subplot(gs[0, 0])      # top-left
    ax_diff = fig.add_subplot(gs[0, 1])      # top-right
    ax_frequency = fig.add_subplot(gs[1,0])
    ax_spatial = fig.add_subplot(gs[1,1])

    # Display the first original frame and the first difference image
    current_orig_im = ax_orig.imshow(original_rgb[0])
    ax_orig.set_title("Original Frame")
    ax_orig.axis("off")

    current_diff_im = ax_diff.imshow(difference_rgb[0])
    ax_diff.set_title("Difference Image (Frame n vs. Frame n+1)")
    ax_diff.axis("off")
    
    current_frequency_img = ax_frequency.imshow(frequency_domain[0])
    ax_frequency.set_title("Frequency Domain Frame")
    ax_frequency.axis("off")
    
    current_spatial_img = ax_spatial.imshow(spatial_domain[0])
    ax_spatial.set_title("Spatial Domain Frame")
    ax_spatial.axis("off")

    # --- Slider to navigate frames manually ---
    # Place the slider below the bottom row
    slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=len(frame_diff) - 1,
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
        current_diff_im.set_data(difference_rgb[idx])
        current_frequency_img.set_data(frequency_domain[idx])
        current_spatial_img.set_data(spatial_domain[idx])
        # Update the vertical line in the plot
        # marker_line.set_xdata([idx, idx])
        # marker_line_1.set_xdata([idx, idx])
        # marker_line_2.set_xdata([idx, idx])
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
            for idx in range(start_idx, len(frame_diff)):
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


def process_frames(original_frames, subtracted_images= [], filtered_images = [], freq_to_spatial_frames = []):
    
    for idx in range(1, len(original_frames)-1):
        
        current_frame  = original_frames[idx]
        next_frame = original_frames[idx+1]
        gs_current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        gs_next_frame = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
                
        kernel = np.ones((5,5),np.float32)/25
        gaussian_blur_curr = cv2.filter2D(gs_current_frame,-1,kernel)
        gaussian_blur_next = cv2.filter2D(gs_next_frame, -1, kernel)
        subtracted_image = cv2.absdiff(gaussian_blur_curr, gaussian_blur_next)
        # subtracted_image = cv2.resize(subtracted_image,(640,480))
        filtered_image, dft_shift = frequency_domain_filtering(subtracted_image)
        filtered_images.append(filtered_image)
        subtracted_images.append(subtracted_image*5)
        
        freq_to_spatial_frame = frequency_to_spatial_domain(subtracted_image, dft_shift)
        freq_to_spatial_frames.append(freq_to_spatial_frame)
    return original_frames, subtracted_images, filtered_images, freq_to_spatial_frames


def crop_frames(frames, cropping, cropped_frames = []):

    dimension = frames[0].shape
    for i in range(1,len(frames)-1):
        cropped_frame = frames[i][36:134, :] 
        cropped_frames.append(cropped_frame)
    if cropping == True:
        return cropped_frames
    return frames

def frequency_domain_filtering(image):
    """
    #Fourier transform and Inverse transform operations using Numpy
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    filtered_image = 20*np.log(np.abs(fshift))
    
    rows,cols = image.shape
    crow,ccol = rows//2, cols//2
    
    fshift[crow-20:crow+21, ccol-20:ccol+21] = 0
    f_ishift = np.fft.ifftshift(fshift)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.real(image_back)
    """
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    epsilon = 1e-8
    magnitude_spectrum = 20 * np.log(magnitude + epsilon)
    
    return magnitude_spectrum, dft_shift

def frequency_to_spatial_domain(image,dft_shift):
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    
    mask = np.zeros((rows, cols, 2), np.uint8)
    # mask[:, ccol-10:ccol+10] = 1
    mask[crow-10:crow+10,:] = 1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return img_back

def main():
    video_path = "data/Queens_bev3.mp4"
    frames = get_frames_from_video(video_path)
    frames = crop_frames(frames,cropping=True)
    original_frames, subtracted_images, filtered_images, freq_to_spatial_frames = process_frames(frames)
    interactive_frames_and_difference_plot(original_frames, subtracted_images, filtered_images, freq_to_spatial_frames)


if __name__ == '__main__':
    main()