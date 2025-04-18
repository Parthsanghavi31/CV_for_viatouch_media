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
import imageio
import numpy as np
from image_enhancement import enhance_image
from PIL import Image, ImageEnhance

def interactive_frames_and_difference_plot(frame_times, frames, frame_diff, canny_edges, subtracted_canny_images, sub_norms, canny_norms, digital_signal, load_sensor_events):
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

    if frames == None or frame_diff == None or canny_edges == None or canny_norms == None:
        print("No frames, subtracted images, or frame_diff found.")
        return

    # Convert original frames (for the relevant indices) to RGB
    # Because frame_diff go from index=1..(len(frames)-1),
    # we'll map each difference image "i" to frames[i+1] or frames[i].
    # For consistency with how process_frames is enumerated, let's use frames[1:-1].
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Convert the grayscale difference images to RGB for Matplotlib display
    difference_rgb = [img for img in frame_diff]
    
    canny_edges_frames = [img for img in canny_edges]
    sub_canny_edges_frames = [img for img in subtracted_canny_images]
   
    #Processing canny_norms
    canny_norm_array = np.array(canny_norms)
    # frame_indices = np.arange(len(frame_times))
    frame_indices = np.array(frame_times)
    
    sub_norm_array = np.array(sub_norms)
    digital_signal_array = np.array(digital_signal)

    # ls_indices = np.arange(load_sensor_events.shape[0])
    
    #Digital Data
    # --- Create the figure and axes layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(10, 2, height_ratios=[20,12,12,1,12,1,12,1,12,1])

    ax_orig = fig.add_subplot(gs[0, 0])      # top-left
    ax_diff = fig.add_subplot(gs[1, 0])      # top-right
    ax_canny_edges_frames = fig.add_subplot(gs[0,1])
    ax_sub_canny_edges_frames = fig.add_subplot(gs[1,1])
    ax_plot_sub_norm = fig.add_subplot(gs[2,:])
    ax_plot_canny_norm = fig.add_subplot(gs[4,:])
    ax_plot_harm = fig.add_subplot(gs[6,:])
    ax_ls_events = fig.add_subplot(gs[8,:])

    # Display the first original frame and the first difference image
    current_orig_im = ax_orig.imshow(original_rgb[0])
    ax_orig.set_title("Original Frame")
    ax_orig.axis("off")

    current_diff_im = ax_diff.imshow(difference_rgb[0])
    ax_diff.set_title("Difference Image (Frame n vs. Frame n+1)")
    ax_diff.axis("off")
    
    current_frequency_img = ax_canny_edges_frames.imshow(canny_edges_frames[0])
    ax_canny_edges_frames.set_title("Canny Edge of Current Frame")
    ax_canny_edges_frames.axis("off")
    
    current_spatial_img = ax_sub_canny_edges_frames.imshow(sub_canny_edges_frames[0])
    ax_sub_canny_edges_frames.set_title("Subtracted image after Canny Edge")
    ax_sub_canny_edges_frames.axis("off")
    
    # Plot the canny_norms on the bottom subplot
    ax_plot_canny_norm.plot(frame_indices, canny_norm_array, label="Canny Frame Difference Norm", color='b')
    ax_plot_canny_norm.set_title("Canny Frame-to-Frame Grayscale Difference")
    ax_plot_canny_norm.set_ylabel("Norm of Difference")
    ax_plot_canny_norm.grid(True)
    ax_plot_canny_norm.legend()
    
    # Plot the sub_norms on the bottom subplot
    ax_plot_sub_norm.plot(frame_indices, sub_norm_array, label="Sub Img Frame Difference Norm", color="red")
    ax_plot_sub_norm.set_title("Frame-to-Frame Grayscale Difference")
    ax_plot_sub_norm.set_ylabel("Norm of Difference")
    ax_plot_sub_norm.grid(True)
    ax_plot_sub_norm.legend()
    
    # Plot the harmonic on the bottom subplot
    ax_plot_harm.plot(frame_indices, digital_signal_array, label="Digital Signal", color='g')
    ax_plot_harm.set_title(" Frame-to-Frame Grayscale Difference")
    ax_plot_harm.set_ylabel("Digital Value ")
    ax_plot_harm.grid(True)
    ax_plot_harm.legend()

    ax_ls_events.plot(frame_indices, load_sensor_events, label ="Load Sensor Events", color = 'g')
    ax_ls_events.set_title("LS Events")
    ax_ls_events.set_xlabel(" Time stamp")
    ax_ls_events.set_ylabel("Type of Events")
    ax_plot_harm.grid(True)
    ax_plot_harm.legend()
    
    marker_line = ax_plot_canny_norm.axvline(x=0, color='r', linestyle='--', linewidth=2)
    marker_line_1 = ax_plot_sub_norm.axvline(x=0, color='r', linestyle='--', linewidth=2)
    marker_line_2 = ax_plot_harm.axvline(x=0, color='r', linestyle='--', linewidth=2)
    marker_line_3 = ax_ls_events.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # --- Slider to navigate frames manually ---
    # Place the slider below the bottom row
    slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=len(frame_diff)-1,
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
        current_frequency_img.set_data(canny_edges_frames[idx])
        current_spatial_img.set_data(sub_canny_edges_frames[idx])
        # Update the vertical line in the plot
        time_x = frame_indices[idx]
        marker_line.set_xdata([time_x, time_x])
        marker_line_1.set_xdata([time_x, time_x])
        marker_line_2.set_xdata([time_x, time_x])
        marker_line_3.set_xdata([time_x, time_x])

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
  
#----------------------------------------------------------------------------------------------------------------------------------------------------# 

def combine_harmonic_mean(scaled_sub, scaled_canny):
    sub_arr = np.array(scaled_sub)
    canny_arr = np.array(scaled_canny)
    numerator = 2.0 * sub_arr * canny_arr
    denominator = sub_arr + canny_arr + 1e-8 
    combined = numerator / denominator
    return combined.tolist()

def get_frames_from_video(video_path, frames = None):
    
    if frames is None:
        frames = []
    original_frames = []
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break    
        # if frame_count>=10:  
         
        resized_frame = cv2.resize(frame, (320,240)) 

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        if np.average(gray) < 45:
            continue
        else:
            frames.append(resized_frame)
            original_frames.append(frame)

        frame_count+=1
        # path = f"frames/frames{frame_count}.jpg"
        # cv2.imwrite(path, frame)
        
    video.release()
    # cv2.destroyAllWindows()
    return frames, original_frames

def process_frames_v2(original_frames, subtracted_images= []):
    canny_images = []
    diff_canny_images = []
    canny_on_sub_images = []
    centroids_per_frame = []

    
    for idx in range(0, len(original_frames)-1):
        
        current_frame  = original_frames[idx]
        next_frame = original_frames[idx+1]
        gs_current_frame = (cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY))
        gs_current_frame_normalized = gs_current_frame/255.0

        gs_next_frame = (cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY))
        gs_next_frame_normalized = gs_next_frame/255.0

        kernel = np.ones((5,5),np.float32)/25
        gaussian_blur_curr = cv2.filter2D(gs_current_frame_normalized,-1,kernel)
        gaussian_blur_next = cv2.filter2D(gs_next_frame_normalized, -1, kernel)
        subtracted_image = cv2.absdiff(gaussian_blur_curr, gaussian_blur_next)
        
        #Applying canny edge on the image difference to ge the object size and texture.
        canny_gaussian_blur_curr = cv2.filter2D(gs_current_frame,-1,kernel)
        canny_gaussian_blur_next = cv2.filter2D(gs_next_frame, -1, kernel)
        canny_subtracted_image = cv2.absdiff(canny_gaussian_blur_curr, canny_gaussian_blur_next)

        # canny_on_sub_img = cv2.Canny(canny_subtracted_image, 10, 200)
        ret, canny_on_sub_img = cv2.threshold(canny_subtracted_image,10,200,cv2.THRESH_BINARY)
        
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        morphed_image =  cv2.morphologyEx(canny_on_sub_img, cv2.MORPH_GRADIENT, cross_kernel, iterations=3)
        contours, hierarchy = cv2.findContours(
                        morphed_image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                        )
        # print(contours)
        black_canvas = np.zeros((morphed_image.shape[0], morphed_image.shape[1], 3), dtype=np.uint8)

        # Set a minimum contour area to filter smaller contours (if required)
        min_area = 800
        filtered_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt)>=min_area:
                filtered_contours.append(cnt)
        # print(filtered_contours)

        # Draw contours onto your black canvas
        cv2.drawContours(black_canvas, filtered_contours, -1, (0, 255, 0), 3)
        
        # Compute centroids for all filtered contours
        centroids = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        # If you want to visualize each contour's centroid and one average centroid:
        if len(centroids) > 0:
            # Draw a small circle (blue) on each individual centroid
            for (cx, cy) in centroids:
                cv2.circle(black_canvas, (cx, cy), 3, (255, 0, 0), -1)

            # Calculate the average of these centroids
            avg_x = int(sum(c[0] for c in centroids) / len(centroids))
            avg_y = int(sum(c[1] for c in centroids) / len(centroids))
            
            # Draw the average centroid in red
            cv2.circle(black_canvas, (avg_x, avg_y), 10, (0, 0, 255), -1)
            cv2.putText(
                black_canvas,
                f"Avg: ({avg_x}, {avg_y})",
                (avg_x + 10, avg_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            avg_y += 70
            centroids_per_frame.append((avg_x,avg_y))
        else:
            centroids_per_frame.append((0,0))
        # print(f"Centroids of Individual frames : {len(centroids_per_frame)} \n")
        canny_on_sub_images.append(black_canvas)
        
        #Appending the image difference, which will be used to calculate the norm
        subtracted_images.append(subtracted_image)
        
       # Canny Edge applied here and then taking the difference of the Canny frames to get a estimate of the object.
        edges_current_frame = cv2.Canny(gs_current_frame, 100, 200)
        canny_images.append(edges_current_frame)
        edges_next_frame = cv2.Canny(gs_next_frame, 100, 200)

        filtered_image = cv2.absdiff(edges_current_frame, edges_next_frame)        
        filtered_image = cv2.medianBlur(filtered_image, 3)        
        diff_canny_images.append(filtered_image)
        
        
    return original_frames, subtracted_images, canny_images, diff_canny_images, canny_on_sub_images, centroids_per_frame

def processing_norms(subtracted_images, diff_canny_images):
    canny_norms = []
    sub_img_norms = []
    for i in range(0, len(diff_canny_images)-1):
        
        sub_norm = np.linalg.norm(subtracted_images[i]*5)
        sub_img_norms.append(sub_norm)
        smoothed_sub_norms = medfilt(np.array(sub_img_norms), kernel_size=3)
        gaussian_sub_norms = gaussian_filter1d(smoothed_sub_norms, sigma=2)
        final_sub_output = list(gaussian_sub_norms)
        
        canny_norm = np.linalg.norm((diff_canny_images[i])/255.0)
        canny_norms.append(canny_norm)
        smoothed_canny_norms  = medfilt(np.array(canny_norms), kernel_size=3)
        gaussian_canny_norms = gaussian_filter1d(smoothed_canny_norms, sigma=2)
        final_canny_output  = list(gaussian_canny_norms)
        
    return final_sub_output, final_canny_output    

def scaled_norms(sub_norms, canny_norms): #input is a list
    
    sub_norm_arr = np.array(sub_norms, dtype = np.float32).reshape(-1,1)
    sub_norm_arr = np.clip(sub_norm_arr, 0, 30)
    canny_norms_arr = np.array(canny_norms, dtype = np.float32).reshape(-1,1)
    canny_norms_arr = np.clip(canny_norms_arr, 0, 15)

    
    scaler = MinMaxScaler(feature_range=(0,1), clip=True)
    scaled_sub_norm = scaler.fit_transform(sub_norm_arr).flatten()
    scaler1 = MinMaxScaler(feature_range=(0,1), clip=True)
    scaled_canny_norm = scaler1.fit_transform(canny_norms_arr).flatten()
    
    combined_harmonic  = combine_harmonic_mean(scaled_sub_norm, scaled_canny_norm)
    #--- Converting Analog Signal to Digital Signal ---#
    threshold = 0.4
    digital_signal = np.where(np.array(combined_harmonic)>threshold, 1, 0).reshape(-1,1)
    
    
    return scaled_sub_norm.tolist(), scaled_canny_norm.tolist(), digital_signal

def pre_processing_digital_signal(digital_signal):

    median_filtered_dg = medfilt(digital_signal.flatten(), kernel_size=7)
    return median_filtered_dg
 
def crop_frames(frames, cropping, cropped_frames = []):

    # dimension = frames[0].shape
    for i in range(0,len(frames)-1):
        cropped_frame = frames[i][70:120, :] 
        cropped_frames.append(cropped_frame)
    if cropping == True:
        return cropped_frames
    return frames

def processing_json_file(frames, start_idx, end_idx, json_file_path, door_messages, user_pickups, user_putbacks):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    activities = data["user_activity_instance"]["user_activities"]
    total_duration = data["duration"]

    n_frames = len(frames)
    frame_times = np.linspace(0, total_duration, num=n_frames)
    clipped_frame_times = frame_times[start_idx:end_idx-2]

    # 1) Parse activity_time to datetime objects
    fmt = "%Y-%m-%d:%H:%M:%S"
    for act in activities:
        act["datetime_obj"] = datetime.strptime(act["activity_time"], fmt)

    # 2) Find earliest and latest time
    min_time = min(act["datetime_obj"] for act in activities)
    max_time = max(act["datetime_obj"] for act in activities)

    # 3) Total time in seconds between earliest and latest
    total_seconds = (max_time - min_time).total_seconds()

    # 4) Scale each activity_time to 0..16
    for act in activities:
        if act['user_activity_type'] == "DOOR_OPENED" or act['user_activity_type'] == "DOOR_CLOSED":
            delta = (act["datetime_obj"] - min_time).total_seconds()
            if total_seconds == 0:
                # If all times are identical (edge case), set everything to zero
                scaled_time = 0
            else:
                scaled_time = total_duration * (delta / total_seconds)
            door_messages.append(int(scaled_time))

        elif act['user_activity_type'] == "USER_PICKUP":
            delta = (act["datetime_obj"] - min_time).total_seconds()
            if total_seconds == 0:
                # If all times are identical (edge case), set everything to zero
                scaled_time = 0
            else:
                scaled_time = total_duration * (delta / total_seconds)
            user_pickups.append(int(scaled_time)-1)
        else:
            delta = (act["datetime_obj"] - min_time).total_seconds()
            if total_seconds == 0:
                # If all times are identical (edge case), set everything to zero
                scaled_time = 0
            else:
                scaled_time = total_duration * (delta / total_seconds)
            user_putbacks.append(int(scaled_time)-1)

    print(door_messages, user_pickups, user_putbacks)
    door_messages[0] = door_messages[0] + 1
    door_messages[1] = door_messages[1] - 2

    load_sensor_events = [-1]*int(total_duration)

    for i in door_messages:
        load_sensor_events[i] = 2
    for i in user_pickups:
        load_sensor_events[i] = 0

    ls_time = np.linspace(0, total_duration, num=len(load_sensor_events)) 

    # 2) Prepare an array to hold the aligned load-sensor events
    ls_events_aligned = np.array([-1] * len(clipped_frame_times))

    # 3) For each event in load_sensor_events, find the closest time in clipped_frame_times.
    for event_idx, event_val in enumerate(load_sensor_events):
        if event_val != -1:
            # Use ls_time[event_idx], not load_sensor_events[event_idx]
            t_event = ls_time[event_idx]                                      
            i = np.argmin(np.abs(clipped_frame_times - t_event))            
            ls_events_aligned[i] = event_val
    return ls_events_aligned, clipped_frame_times

def user_act_digi_signal(crucial_indices, centroids_per_frame):
    all_activities = []
    processed_centroids = []

    ind_user_act = []
    ind_centroids_list = []

    for i, value in enumerate(crucial_indices):
        centroid = centroids_per_frame[i]

        if i == 0:
            ind_user_act.append(value)
            ind_centroids_list.append(centroid)
        else:
            if value - crucial_indices[i - 1] > 1:
                # Save previous group
                all_activities.append(ind_user_act)
                
                # Process average centroid for previous group
                valid_centroids = [pt for pt in ind_centroids_list if tuple(pt) != (0, 0)]
                if valid_centroids:
                    avg_x = int(sum(pt[0] for pt in valid_centroids) / len(valid_centroids))
                    avg_y = int(sum(pt[1] for pt in valid_centroids) / len(valid_centroids))
                    avg_centroid = (avg_x, avg_y)
                else:
                    avg_centroid = (0, 0)

                processed_centroids.append([avg_centroid] * len(ind_centroids_list))

                # Reset for next group
                ind_user_act = []
                ind_centroids_list = []

            # Continue building current group
            ind_user_act.append(value)
            ind_centroids_list.append(centroid)

    # Process final group
    if ind_user_act:
        all_activities.append(ind_user_act)

        valid_centroids = [pt for pt in ind_centroids_list if tuple(pt) != (0, 0)]
        if valid_centroids:
            avg_x = int(sum(pt[0] for pt in valid_centroids) / len(valid_centroids))
            avg_y = int(sum(pt[1] for pt in valid_centroids) / len(valid_centroids))
            avg_centroid = (avg_x, avg_y)
        else:
            avg_centroid = (0, 0)

        processed_centroids.append([avg_centroid] * len(ind_centroids_list))

    return all_activities, processed_centroids

def draw_bbox_original_frame(org_frames, original_frames, grouped_indices, grouped_centroids):
    # Flatten index-centroid pairs for quick lookup
    frame_to_centroid = {}
    modified_frames = []
    bbox_cropped_frames = []
    for group_indices, group_centroids in zip(grouped_indices, grouped_centroids):
        for idx, centroid in zip(group_indices, group_centroids):
            frame_to_centroid[idx] = centroid

    # Loop through all frames
    for i, frame in enumerate(org_frames):
        frame_copy = frame.copy()
        # pad_img = frame.copy()
        original_frame_copy = original_frames[i].copy()
        # Draw box only if this frame has a valid centroid
        if i in frame_to_centroid:
            avg_x, avg_y = frame_to_centroid[i]

            if (avg_x, avg_y) != (0, 0):
                cv2.circle(frame_copy, (avg_x, avg_y), 10, (0, 0, 255), -1)
                pt1 = (avg_x - 90, avg_y - 30)
                pt2 = (avg_x + 70, avg_y + 80)
                cv2.rectangle(frame_copy, pt1, pt2, (0, 255, 0), 3)
                pad_img = np.pad(original_frame_copy, ((180,180), (180,180), (0,0)), mode='wrap')
                cropped_frame = pad_img[(pt1[1]+90)*2:(pt2[1]+90)*2, (pt1[0]+90)*2:(pt2[0]+90)*2]
                bbox_cropped_frames.append(cropped_frame)
                # print(cropped_frame.shape)
                # cv2.imshow("ROI_frames", cropped_frame)
                # key = cv2.waitKey(100)
                # if key == 27:  # ESC to exit
                #     break


        modified_frames.append(frame_copy)
    # cv2.destroyAllWindows()
    return modified_frames, bbox_cropped_frames

def apply_pil_enhancements(img, brightness, contrast, color, sharpness):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def image_enhancement(frames, enhanced_frames, frame_save_path,
                      brightness=0.9, contrast=1.2, color=1.6, sharpness=2.0):
    
    for i, frame in enumerate(frames):
        # --- PIL Enhancements ---
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_pil = apply_pil_enhancements(img_pil, brightness, contrast, color, sharpness)
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
        frame_name = f"frame_{i}.jpg"
        cv2.imwrite(os.path.join(frame_save_path, frame_name), image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        enhanced_frames.append(image)
    print("Completed Saving Images")
    return enhanced_frames
 
#----------------------------------------------------------------------------------------------------------------------------------------------------# 
        
def main():
    transaction_number = "office_transaction_10"

    
    enhanced_frames = []
    door_messages = []
    user_pickups = []
    user_putbacks = []
    transaction = os.path.join("data_for_Mapping_logic", transaction_number)
    json_file_path = os.path.join(transaction, "user_activites.json")
    video_path = os.path.join(transaction, "media.mp4")
    
    frame_save_path = os.path.join("frames_for_classification", transaction_number)
    os.makedirs(frame_save_path, exist_ok=True)
    
    frames, original_frames = get_frames_from_video(video_path)
    cropped_frames = crop_frames(frames,cropping=True)
    start_idx = 10
    end_idx   = len(cropped_frames) - 10
    if end_idx <= start_idx:
        print("Not enough frames after cropping; adjust start/end indices.")
        return
    
    ls_events_aligned, clipped_frame_times = processing_json_file(cropped_frames, start_idx, end_idx, json_file_path, door_messages, user_pickups, user_putbacks)
    original_frames = original_frames[start_idx:end_idx]
    valid_frames = frames[start_idx:end_idx]
    valid_cropped_frames = cropped_frames[start_idx:end_idx]
    _, subtracted_images, canny_images, diff_canny_images, canny_on_sub_images, centroids_per_frame = process_frames_v2(valid_cropped_frames)
    sub_norms, canny_norms = processing_norms(subtracted_images, diff_canny_images)
    
    scaled_sub_norm, scaled_canny_norm, digital_signal = scaled_norms(sub_norms, canny_norms)
    processed_digital_signal = pre_processing_digital_signal(digital_signal)
    crucial_indices = np.where(digital_signal>0)[0]

    selected_centroids = np.array(centroids_per_frame)[crucial_indices]
    
    all_activities, processed_centroids = user_act_digi_signal(crucial_indices, selected_centroids)
        
    modified_frames, bbox_cropped_frames = draw_bbox_original_frame(valid_frames, original_frames, all_activities[1:-1], processed_centroids[1:-1])
    enhanced_frames = image_enhancement(bbox_cropped_frames, enhanced_frames, frame_save_path)
    
    # interactive_frames_and_difference_plot(clipped_frame_times, modified_frames, canny_on_sub_images, canny_images, diff_canny_images, scaled_sub_norm, scaled_canny_norm, processed_digital_signal, ls_events_aligned)
    
if __name__ == '__main__':
    main()