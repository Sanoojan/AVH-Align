import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import pandas as pd
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

class FeatureDataset(IterableDataset):
    def __init__(self, metadata_path, features_root_path, tau=15, audio_dim=1024):
        super().__init__()
        self.tau = tau
        self.features_root_path = features_root_path
        self.audio_dim = audio_dim
        self.metadata = pd.read_csv(metadata_path)
        self.num_videos = len(self.metadata)
        self.num_frames = [self.metadata.iloc[i]['num_frames'] for i in range(len(self.metadata))]
        
        self.paths = self.metadata['path'].apply(lambda x: os.path.join(self.features_root_path, x.replace(".mp4", ".npz"))).tolist()

    def __len__(self):
        return sum(self.num_frames)  # Total number of frames across all videos

    def _load_temporal_window(self, audio_feat, video_idx, frame_idx):
        num_frames = self.num_frames[video_idx]
        audio_window = []
        
        for t in range(frame_idx - self.tau, frame_idx + self.tau + 1):
            if 0 <= t < num_frames:
                audio_feature = torch.from_numpy(audio_feat[t]).float()
                audio_feature = audio_feature / (torch.linalg.norm(audio_feature, ord=2, dim=-1, keepdim=True))
            else:
                audio_feature = torch.zeros(self.audio_dim)
            audio_window.append(audio_feature)
            
        return torch.stack(audio_window).float()

    def _load_features(self, video_idx):
        feature = np.load(self.paths[video_idx], allow_pickle=True, mmap_mode='r')
        # breakpoint()
        visual, multimodal = feature['visual_map'], feature['multimodal'] # changed
        if len(visual.shape) == 3:
            # take the diagonal of the visual map
            visual = np.array([visual[i, i] for i in range(visual.shape[0])])
        return visual, multimodal
    

    def _get_worker_videos(self):
        """
        Multi-worker data partition
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker case (no multiprocessing)
            return range(self.num_videos)
        
        # Multi-worker case: Split videos across workers
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        return range(worker_id, self.num_videos, num_workers)

    def __iter__(self):
        worker_videos = self._get_worker_videos()
        
        for video_idx in worker_videos:
            try:
                feature_visual, feature_audio = self._load_features(video_idx)
                num_frames = len(feature_visual)

                for local_frame_idx in range(num_frames):
                    visual_tensor = torch.from_numpy(feature_visual[local_frame_idx]).float()
                    visual_tensor = visual_tensor / (torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True))
                    
                    # extract audio neighborhood centered at local_frame_idx
                    audio_tensor = self._load_temporal_window(feature_audio, video_idx, local_frame_idx)
                    
                    yield visual_tensor, audio_tensor, video_idx, local_frame_idx
            except Exception as e:
                print(f"Error loading video {video_idx}: {e}")
                


from torch.utils.data import IterableDataset, get_worker_info

class AVFeatureDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        features_root_path,
        T_train=100,
        debug=False,
        synthesize_at_feature=False,
        synthesize_prob=0.5,
        synthstyle="random",
        hard_min_start_deviation=10,
        hard_max_start_deviation=30,
        max_fake_segment_length=10,
        num_max_synth_segments_per_video=3,
        synth_modalities="random",
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.features_root_path = features_root_path
        self.T_train = T_train
        self.debug = debug
        self.synthesize_at_feature = bool(synthesize_at_feature)
        self.synthesize_prob = min(1.0, max(0.0, float(synthesize_prob)))
        self.synthstyle = synthstyle
        self.hard_min_start_deviation = int(hard_min_start_deviation)
        self.hard_max_start_deviation = int(hard_max_start_deviation)
        self.max_fake_segment_length = int(max_fake_segment_length)
        self.num_max_synth_segments_per_video = int(num_max_synth_segments_per_video)
        self.synth_modalities = synth_modalities

        if self.synthstyle not in {"random", "hard"}:
            raise ValueError(f"Unknown synthstyle '{self.synthstyle}'. Expected 'random' or 'hard'.")
        if self.synth_modalities not in {"random", "audio", "video", "both"}:
            raise ValueError(
                f"Unknown synth_modalities '{self.synth_modalities}'. "
                "Expected 'random', 'audio', 'video', or 'both'."
            )

        self.paths = self.metadata['path'].apply(
            lambda x: os.path.join(features_root_path, x.replace(".mp4", ".npz"))
        ).tolist()

        self.label_paths = [p.replace(".npz", "_labels.npz") for p in self.paths]
        self.videowise_labels = self.metadata['label'].tolist()

        if self.debug:
            print("Total videos:", len(self.paths))
            print("Real videos:", sum(np.array(self.videowise_labels) == 0))
            print("Fake videos:", sum(np.array(self.videowise_labels) == 1))
            print("Synthesize at feature:", self.synthesize_at_feature)

    def __len__(self):
        return len(self.paths)

    def _generate_fake_segments(self, video_len_frames):
        if video_len_frames <= 1 or self.num_max_synth_segments_per_video <= 0:
            return [], np.zeros(video_len_frames, dtype=np.float32)

        max_seg_len = max(1, min(self.max_fake_segment_length, video_len_frames - 1))
        max_segments = max(1, self.num_max_synth_segments_per_video)
        num_segments = np.random.randint(1, max_segments + 1)
        segments = []
        attempts = 0
        max_attempts = 50

        while len(segments) < num_segments and attempts < max_attempts:
            attempts += 1
            seg_len = np.random.randint(1, max_seg_len + 1)
            start = np.random.randint(0, video_len_frames - seg_len + 1)
            end = start + seg_len

            if any(not (end <= existing_start or start >= existing_end) for existing_start, existing_end in segments):
                continue
            segments.append((start, end))

        labels = np.zeros(video_len_frames, dtype=np.float32)
        for start, end in segments:
            labels[start:end] = 1.0
        return segments, labels

    @staticmethod
    def _choose_random_replacement_start(start, seg_len, total_len):
        max_start = total_len - seg_len
        if max_start <= 0:
            return 0

        replacement_start = np.random.randint(0, max_start + 1)
        for _ in range(10):
            candidate = np.random.randint(0, max_start + 1)
            if abs(candidate - start) > seg_len:
                replacement_start = candidate
                break
        return int(replacement_start)

    def _choose_hard_replacement_start(self, start, seg_len, total_len):
        max_start = total_len - seg_len
        if max_start <= 0:
            return None

        min_deviation = max(0, self.hard_min_start_deviation)
        max_deviation = max(min_deviation, self.hard_max_start_deviation)
        candidates = [
            candidate
            for candidate in range(max_start + 1)
            if min_deviation <= abs(candidate - start) <= max_deviation
        ]
        if not candidates:
            return None

        non_overlapping = [
            candidate
            for candidate in candidates
            if candidate + seg_len <= start or candidate >= start + seg_len
        ]
        candidates = non_overlapping or candidates
        return int(np.random.choice(candidates))

    def _choose_replacement_start(self, start, seg_len, total_len):
        if self.synthstyle == "hard":
            replacement_start = self._choose_hard_replacement_start(start, seg_len, total_len)
            if replacement_start is not None:
                return replacement_start
        return self._choose_random_replacement_start(start, seg_len, total_len)

    def _choose_synth_modality(self):
        if self.synth_modalities == "random":
            return str(np.random.choice(["audio", "video", "both"]))
        return self.synth_modalities

    def _apply_feature_synthesis(self, visual, audio):
        if not self.synthesize_at_feature or np.random.random() >= self.synthesize_prob:
            return visual, audio, np.zeros(len(visual), dtype=np.float32)

        visual = np.array(visual, copy=True)
        audio = np.array(audio, copy=True)
        segments, labels = self._generate_fake_segments(len(visual))
        modality = self._choose_synth_modality()
        total_len = len(visual)

        for start, end in segments:
            seg_len = end - start
            if seg_len <= 0 or seg_len >= total_len:
                labels[start:end] = 0.0
                continue

            replacement_start = self._choose_replacement_start(start, seg_len, total_len)
            replacement_end = replacement_start + seg_len

            if modality in {"audio", "both"}:
                audio[start:end] = audio[replacement_start:replacement_end]
            if modality in {"video", "both"}:
                visual[start:end] = visual[replacement_start:replacement_end]

        return visual, audio, labels

    def __getitem__(self, video_idx):

        feature = np.load(self.paths[video_idx])

        visual_full = feature['visual']
        audio_full = feature['audio']
        multimodal_full = feature['multimodal']

        T = len(visual_full)
        T_train = self.T_train

        # --------------------------------------------------
        # Load framewise labels
        # --------------------------------------------------
        if self.videowise_labels[video_idx] == 0:
            full_labels = np.zeros(T)
        else:
            label_data = np.load(self.label_paths[video_idx])
            full_labels = label_data['framewise_labels']

            # 🚨 If fake video has ZERO fake frames → log & fix
            if full_labels.sum() == 0:
                print(f"[WARNING] Fake video but no fake frames: {self.paths[video_idx]}")
                full_labels = np.zeros(T)  # treat as real

        # --------------------------------------------------
        # Sampling
        # --------------------------------------------------
        if T >= T_train:

            if self.videowise_labels[video_idx] == 1 and full_labels.sum() > 0:

                # Try random windows until at least one fake frame appears
                max_trials = 10
                found = False

                for _ in range(max_trials):
                    start = np.random.randint(0, T - T_train + 1)
                    end = start + T_train
                    if full_labels[start:end].sum() > 0:
                        found = True
                        break

                # If not found after trials, fallback to random window
                if not found:
                    start = np.random.randint(0, T - T_train + 1)
                    end = start + T_train

            else:
                start = np.random.randint(0, T - T_train + 1)
                end = start + T_train

            visual = visual_full[start:end]
            audio = audio_full[start:end]
            multimodal = multimodal_full[start:end]
            labels = full_labels[start:end]

        else:
            repeat_factor = int(np.ceil(T_train / T))

            visual = np.tile(visual_full, (repeat_factor, 1))[:T_train]
            audio = np.tile(audio_full, (repeat_factor, 1))[:T_train]
            multimodal = np.tile(multimodal_full, (repeat_factor, 1))[:T_train]
            labels = np.tile(full_labels, repeat_factor)[:T_train]

        # --------------------------------------------------
        # Optional train-time synthesis for real videos
        # --------------------------------------------------
        if self.videowise_labels[video_idx] == 0:
            visual, audio, synth_labels = self._apply_feature_synthesis(visual, audio)
            labels = synth_labels

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------
        visual = torch.from_numpy(np.array(visual, copy=True)).float()
        audio = torch.from_numpy(np.array(audio, copy=True)).float()
        multimodal = torch.from_numpy(np.array(multimodal, copy=True)).float()
        labels = torch.from_numpy(np.array(labels, copy=True)).float()

        # Normalize
        visual = F.normalize(visual, dim=-1)
        audio = F.normalize(audio, dim=-1)
        multimodal = F.normalize(multimodal, dim=-1)

        # features = torch.stack([visual, audio, multimodal], dim=0)
        features = torch.stack([visual, audio], dim=0)
        

        return features, labels