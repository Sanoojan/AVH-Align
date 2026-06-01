import cv2

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    # Sometimes FPS metadata is wrong or zero
    if fps == 0 or fps is None:
        raise ValueError("FPS metadata not available.")

    return fps, int(frame_count)

video_path="/egr/research-sprintai/baliahsa/projects/AVH-Align/data/DeepfakeDatasets/AV_Deepfake1M/AV-Deepfake1M-PlusPlus/val/val/vox_celeb_2/id09078/hoNjK-fMOK8/00116/fake_video_fake_audio_p1.mp4"
fps, total_frames = get_video_fps(video_path)
print("FPS:", fps)
print("Total Frames:", total_frames)