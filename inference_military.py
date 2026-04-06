import os
import sys
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from ultralytics import RTDETR


# --- DYNAMIC PROJECT LOADER ---
def load_specific_project(project_path):
    """Loads build_model and config from Uniform3 (EfficientNet architecture)"""
    project_path = str(Path(project_path).resolve())
    if project_path not in sys.path:
        sys.path.insert(0, project_path)

    # Force reload to avoid ResNet50/EfficientNet shape mismatch
    for mod in ['models.model', 'configs.config']:
        if mod in sys.modules:
            del sys.modules[mod]

    import configs.config as cfg
    from models.model import build_model
    return build_model, cfg


# --- Global Config ---
STOP_TIME_SEC = 144  # Stop at 2 minutes 24 seconds
SMOOTH_WINDOW = 15  # Stability window for marchers
DISTANCE_THRESHOLD = 0.65  # Sensitivity for uniform matching
TOP_K = 3  # Neighbors to check in gallery
MIN_BOX_HEIGHT = 160  # Only process soldiers in the foreground


class ParadeTracker:
    def __init__(self, window=SMOOTH_WINDOW):
        self.window = window
        self.history = {}

    def update(self, track_id, class_name):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window)
        self.history[track_id].append(class_name)
        return Counter(self.history[track_id]).most_common(1)[0][0]


def get_embedding(model, cv2_img, device, preprocess):
    if cv2_img is None or cv2_img.size == 0: return None
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).cpu().numpy().flatten()


def calculate_top_k_dist(emb, gallery_embs):
    dists = [np.linalg.norm(emb - g) for g in gallery_embs]
    dists.sort()
    return np.mean(dists[:min(TOP_K, len(dists))])


def run_military_uniform3(project_root, checkpoint_name, video_path, samples_root):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. LOAD UNIFORM3 ARCHITECTURE
    build_model, cfg = load_specific_project(project_root)

    preprocess = T.Compose([
        T.Resize(cfg.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD)
    ])

    # 2. LOAD MODEL & WEIGHTS
    model = build_model(device=device)
    checkpoint_path = os.path.join(project_root, "checkpoints", checkpoint_name)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    weights = state["model_state"] if "model_state" in state else state
    model.load_state_dict(weights, strict=False)
    model.eval()
    print(f"--- Military Model Loaded: EfficientNet-B0 (Uniform3) ---")

    detector = RTDETR("rtdetr-l.pt")
    tracker = ParadeTracker()

    # 3. Build Country Galleries
    class_galleries = {}
    colors = {
        "China": (0, 0, 255), "India": (0, 165, 255), "Indonesia": (255, 0, 0),
        "Japan": (255, 255, 255), "Mongolia": (0, 255, 0), "Philippines": (255, 255, 0),
        "UNCERTAIN": (150, 150, 150)
    }

    print(f"--- Building Galleries from {samples_root} ---")
    for country_folder in Path(samples_root).iterdir():
        if country_folder.is_dir():
            name = country_folder.name
            embs = []
            for img_p in country_folder.glob("*"):
                img = cv2.imread(str(img_p))
                e = get_embedding(model, img, device, preprocess)
                if e is not None: embs.append(e)
            if embs:
                class_galleries[name] = embs
                print(f"Loaded {len(embs)} samples for {name}")

    # 4. Process Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    win_name = "Military Parade (Uniform3 - EfficientNet)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        current_sec = frame_count / fps
        if current_sec > STOP_TIME_SEC:
            print("Reached 2:24 limit. Exiting.")
            break

        results = detector.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                # Filter small background boxes
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue

                if frame_count % 3 == 0:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    emb = get_embedding(model, crop, device, preprocess)

                    if emb is not None:
                        scores = {n: calculate_top_k_dist(emb, g) for n, g in class_galleries.items()}
                        best_match = min(scores, key=scores.get)

                        label = best_match if scores[best_match] < DISTANCE_THRESHOLD else "UNCERTAIN"
                        final_label = tracker.update(track_id, label)
                else:
                    final_label = tracker.history.get(track_id, ["Analyzing..."])[-1]

                color = colors.get(final_label, (100, 100, 100))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{final_label.upper()}"
                (w, h), _ = cv2.getTextSize(label_text, 0, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 5), 0, 0.5, (255, 255, 255), 1)

        # HUD: Progress Bar
        progress_w = int((current_sec / STOP_TIME_SEC) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0] - 5), (progress_w, frame.shape[0]), (0, 255, 0), -1)

        cv2.imshow(win_name, frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_military_uniform3(
        project_root="/home/dior/Projects/Uniform3",
        checkpoint_name="best.pt",
        video_path="/home/dior/Projects/Uniform2/videos/military.mp4",
        samples_root="/home/dior/Projects/Uniform2/samples/military"
    )