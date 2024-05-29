# Software created by Hack2Act
# Licence GNU GPU 3
# Version 0.1
# This software import q mp4 video and provide some analysis of positions of some skeleton point of interest in order to help analyzing the quality of the walking of amputees. 

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose model with detailed parameters.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the landmarks of interest and pairs to be analyzed.
landmark_pairs = {
    'Shoulders': (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    'Hips': (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    'Ankles': (mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value)
}

# To store differences of y-values.
y_differences = {key: [] for key in landmark_pairs}

# Load the video for processing.
video_path = 'testmarchenormalefacejuju.mp4'
video_name = video_path.split('/')[-1].split('.')[0]  # Extract base name for output files
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer for output videos.
out_original = cv2.VideoWriter(f'{video_name}_original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
out_skeleton = cv2.VideoWriter(f'{video_name}_skeleton.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save original frame to output video.
    out_original.write(frame)

    # Convert the frame color to RGB since MediaPipe uses RGB images.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Process each frame for the specified landmarks.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        for key, (left_idx, right_idx) in landmark_pairs.items():
            left_point = int(landmarks[left_idx].y * frame.shape[0])
            right_point = int(landmarks[right_idx].y * frame.shape[0])
            y_differences[key].append(left_point - right_point)  # Calculate and store the difference.

    # Save skeleton frame to output video.
    out_skeleton.write(frame)
    cv2.imshow('Pose Detection with Skeleton', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out_original.release()
out_skeleton.release()
cv2.destroyAllWindows()

# FFT analysis setup.
def plot_fft(differences, title, ax, filename):
    """ Calculate and plot the FFT power spectrum. """
    fft_values = np.fft.fft(differences)
    frequencies = np.fft.fftfreq(len(fft_values), d=1 / fps)  # Correctly use frame rate for frequency calculation
    power = np.abs(fft_values) ** 2
    ax.plot(frequencies[:len(frequencies)//2], power[:len(power)//2])  # Plot only positive frequencies
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    plt.savefig(filename)

# Plotting the differences and FFT results.
fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Differences in Y-Position and Frequency Analysis')

# Plot each difference in y-values and save figures.
for ax, (key, values) in zip(axs[:3], y_differences.items()):
    ax.plot(values, label=f'Difference between {key}')
    ax.set_title(f'{key} Y-Position Differences')
    ax.set_ylabel('Difference in Y-Position (pixels)')
    ax.legend()

plot_fft(y_differences['Shoulders'], 'Power Spectrum of Shoulder Movement Differences', axs[3], f'{video_name}_shoulders_fft.jpeg')
plot_fft(y_differences['Ankles'], 'Power Spectrum of Ankle Movement Differences', axs[4], f'{video_name}_ankles_fft.jpeg')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
