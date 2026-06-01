import numpy as np
import torch

feature_path="Features/AV1M-Trimmed_with_random/train/id00012/21Uxsk56VDQ/00001/real.npz"

data = np.load(feature_path, allow_pickle=True)

# save_dict = {
#             "visual": feature_vid,
#             "audio": feature_audio,
#             "multimodal": feature_multimodal,
#             "audio_random": feature_audio_random,
#             "video_random": feature_video_random,
#             "audio_indices": audio_indices,
#             "video_indices": video_indices,
#         }

audio = data["audio"]
visual = data["visual"]
multimodal = data["multimodal"]
audio_random = data["audio_random"]
video_random = data["video_random"]
audio_indices = data["audio_indices"]
video_indices = data["video_indices"]

audio = audio / np.linalg.norm(audio, axis=-1, keepdims=True)
visual = visual / np.linalg.norm(visual, axis=-1, keepdims=True)
multimodal = multimodal / np.linalg.norm(multimodal, axis=-1, keepdims=True)
audio_random = audio_random / np.linalg.norm(audio_random, axis=-1, keepdims=True)
video_random = video_random / np.linalg.norm(video_random, axis=-1, keepdims=True)


audio_visual_similarity = np.einsum('ij,ij->i', visual, audio)
audio_multimodal_similarity = np.einsum('ij,ij->i', multimodal, audio)
visual_multimodal_similarity = np.einsum('ij,ij->i', visual, multimodal)

audio_visual_similarity_random = np.einsum('ij,ij->i', video_random, audio_random)
audio_random_multimodal_similarity_random = np.einsum('ij,ij->i', multimodal, audio_random)
visual_random_multimodal_similarity_random = np.einsum('ij,ij->i', multimodal, video_random)

framewise_scores = {
    "audio_visual_similarity": audio_visual_similarity,
    "audio_multimodal_similarity": audio_multimodal_similarity,
    "visual_multimodal_similarity": visual_multimodal_similarity,
    "audio_visual_similarity_random": audio_visual_similarity_random,
    "audio_random_multimodal_similarity_random": audio_random_multimodal_similarity_random,
    "visual_random_multimodal_similarity_random": visual_random_multimodal_similarity_random
}

# Total number of frames (should match similarity length)
num_frames = len(visual_random_multimodal_similarity_random)

# Initialize zero mask
video_mask = np.zeros(num_frames, dtype=int)

# Set selected indices to 1
video_mask[video_indices] = 1


mean_std_scores = {}
for key, scores in framewise_scores.items():
    mean = np.mean(scores)
    std = np.std(scores)
    mean_std_scores[key] = (mean, std)

print("Framewise Similarity Scores (mean ± std):")
for key, (mean, std) in mean_std_scores.items():
    print(f"{key}: {mean:.4f} ± {std:.4f}")


# breakpoint()
# plot visual_random_multimodal_similarity_random vs video_indices
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(video_mask, visual_random_multimodal_similarity_random, label='Visual Random vs Multimodal', marker='o')
plt.xlabel('Video Indices')
plt.ylabel('Similarity Score')
plt.title('Visual Random vs Multimodal Similarity Scores')
plt.legend()
plt.grid()
plt.savefig("visual_random_multimodal_similarity_random.png")


import matplotlib.pyplot as plt
import numpy as np

# X axis = frame numbers
frames = np.arange(num_frames)

plt.figure(figsize=(10, 6))


plt.plot(frames,
         audio_random_multimodal_similarity_random,
         label='Similarity',
         linewidth=2)

plt.scatter(video_indices,
            audio_random_multimodal_similarity_random[video_indices],
            marker='x',
            s=80,
            label='Selected Frames')



non_masked_indices = np.setdiff1d(frames, video_indices)

plt.scatter(non_masked_indices,
            audio_random_multimodal_similarity_random[non_masked_indices],
            marker='o',
            s=60,
            label='Unmasked Frames')

plt.xlabel('Frame Index')
plt.ylabel('Similarity Score')
plt.title('Audio Random vs Multimodal Similarity')
plt.legend()
plt.grid()
plt.savefig("audio_random_multimodal_similarity_random2.png")
plt.show()

print("done")


# breakpoint()
print("done")

