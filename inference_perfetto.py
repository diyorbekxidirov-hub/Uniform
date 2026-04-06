import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from ultralytics import RTDETRq
from collections import Counter
import time
import os

from configs.config import IMAGE_SIZE, NORM_MEAN, NORM_STD
from models.model import build_model

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — tune these after running calibrationSS
# ──────────────────────────────────────────────────────────────────────────────
WAITER_THRESHOLD = 0.43  # below → WAITER
UNCERTAIN_THRESHOLD = 0.55  # between → UNCERTAIN, above → GUEST
TOP_K_GALLERY = 5  # use avg of 5 closest gallery matches
VOTE_FRAMES = 50  # frames before locking a decision (was 30)
EMBED_EVERY_N = 2  # run embedding every N frames (was 4)
START_OFFSET_SEC = (0 * 3600) + (12 * 60) + (34) # 1h34m — set to 0 to start from beginning

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────
ignore_pts = []
inference_started = False
video_writer = None  # Added for recording

PREPROCESS = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


# ──────────────────────────────────────────────────────────────────────────────
# Stable Tracker — votes over N frames before locking
# ──────────────────────────────────────────────────────────────────────────────

class StableTracker:
    def __init__(self, vote_frames: int = VOTE_FRAMES):
        self.vote_frames = vote_frames
        self.history = {}  # track_id -> list of (name, color)
        self.frames_seen = {}  # track_id -> int
        self.locked = {}  # track_id -> (name, color)  — final decision

    def update(self, track_id: int, current_label_info):
        # Already locked — return immediately, no re-evaluation
        if track_id in self.locked:
            return self.locked[track_id]

        self.frames_seen[track_id] = self.frames_seen.get(track_id, 0) + 1

        # Only record confident labels (not UNCERTAIN or None)
        if current_label_info and current_label_info[0] not in ("UNCERTAIN", "Analyzing..."):
            self.history.setdefault(track_id, []).append(current_label_info)

        count = self.frames_seen[track_id]
        history = self.history.get(track_id, [])

        if count >= self.vote_frames and len(history) > 0:
            votes = Counter([h[0] for h in history])
            winner, win_count = votes.most_common(1)[0]
            confidence = win_count / len(history)

            # Only lock if confident enough (>60% of votes agree)
            if confidence >= 0.6:
                if winner == "WAITER":
                    # Waiters are NOT locked — they can move in/out of frame
                    # Reset counter so we keep re-evaluating
                    self.frames_seen[track_id] = 0
                    self.history[track_id] = []
                    return None
                else:
                    # Lock guests — they don't become waiters
                    self.locked[track_id] = ("GUEST", (0, 0, 255))
                    return self.locked[track_id]

        return None

    def cleanup(self, active_ids: set):
        """Remove stale track IDs that are no longer in frame."""
        stale = [tid for tid in self.frames_seen if tid not in active_ids]
        for tid in stale:
            self.history.pop(tid, None)
            self.frames_seen.pop(tid, None)
            # Keep locked — they may re-enter frame


# ──────────────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding(model, cv2_img, device):
    if cv2_img is None or cv2_img.size == 0:
        return None
    if cv2_img.shape[0] < 10 or cv2_img.shape[1] < 10:
        return None
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).cpu().numpy().flatten()


def gallery_distance(emb: np.ndarray, gallery: list, top_k: int = TOP_K_GALLERY) -> float:
    """
    Average L2 distance to the top-k closest gallery embeddings.
    More robust than nearest-neighbour — one bad gallery image
    doesn't dominate the decision.
    """
    dists = sorted([np.linalg.norm(emb - g) for g in gallery])
    k = min(top_k, len(dists))
    return float(np.mean(dists[:k]))


def classify(dist: float):
    """Returns (label, color) based on distance."""
    if dist < WAITER_THRESHOLD:
        return "WAITER", (0, 220, 0)
    elif dist <= UNCERTAIN_THRESHOLD:
        return "UNCERTAIN", (0, 220, 220)
    else:
        return "GUEST", (0, 0, 220)


# ──────────────────────────────────────────────────────────────────────────────
# Mouse callback — draw ignore zone before starting
# ──────────────────────────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global ignore_pts
    if not inference_started:
        if event == cv2.EVENT_LBUTTONDOWN:
            ignore_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            ignore_pts.clear()


def in_ignore_zone(center) -> bool:
    if len(ignore_pts) < 3:
        return False
    return cv2.pointPolygonTest(np.array(ignore_pts), center, False) >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Gallery calibration — prints recommended thresholds
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(gallery: list):
    if len(gallery) < 2:
        print("[Calibrate] Not enough gallery images to calibrate.")
        return
    dists = [np.linalg.norm(gallery[i] - gallery[j])
             for i in range(len(gallery))
             for j in range(i + 1, len(gallery))]
    print(f"\n[Calibrate] Waiter-Waiter distances across {len(gallery)} gallery images:")
    print(f"  min={np.min(dists):.3f}  avg={np.mean(dists):.3f}  max={np.max(dists):.3f}")
    suggested_waiter = np.max(dists) * 1.2
    suggested_uncertain = np.max(dists) * 1.5
    print(f"  Suggested WAITER_THRESHOLD:    {suggested_waiter:.3f}")
    print(f"  Suggested UNCERTAIN_THRESHOLD: {suggested_uncertain:.3f}")
    print(f"  (update the CONFIG section at the top of this file)\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(checkpoint, video_path, samples):
    global inference_started, video_writer

    if not os.path.exists(video_path):
        print(f"[Error] Video not found: {video_path}")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device.upper()}")

    model = build_model(device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(
        state["model_state"] if "model_state" in state else state,
        strict=False
    )
    model.eval()
    epoch = state.get("epoch", "?") if isinstance(state, dict) else "?"
    print(f"[Model] Loaded checkpoint: {checkpoint}  (epoch {epoch})")

    # ── Gallery ───────────────────────────────────────────────────────────────
    gallery = []
    for p in sorted(Path(samples).rglob("*")):
        if p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            img = cv2.imread(str(p))
            emb = get_embedding(model, img, device)
            if emb is not None:
                gallery.append(emb)
    print(f"[Gallery] {len(gallery)} reference embeddings loaded.")

    # Print calibration info so you know if thresholds need adjusting
    calibrate(gallery)

    # ── Detector ──────────────────────────────────────────────────────────────
    detector = RTDETR("rtdetr-l.pt")
    print("[Detector] RT-DETR ready.")

    # ── Video ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return

    if START_OFFSET_SEC > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, START_OFFSET_SEC * 1000)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_frame_duration = 1.0 / (native_fps * 0.8)  # 0.8x playback speed

    # ── Window ────────────────────────────────────────────────────────────────
    win_name = "Uniform Detection | S: start  Q: quit  LClick: ignore zone  RClick: clear"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, mouse_callback)

    stable_tracker = StableTracker(vote_frames=VOTE_FRAMES)

    ret, setup_frame = cap.read()
    if not ret:
        print("[Error] Cannot read first frame.")
        return

    frame_count = 0
    last_embeddings = {}  # track_id -> (name, color, dist) from last embedding run

    print("\n[Ready] Draw ignore zone with left-click, right-click to clear.")
    print("        Press S to start inference, Q to quit.\n")

    while True:
        loop_start = time.perf_counter()

        # ── Setup mode — draw ignore zone ────────────────────────────────────
        if not inference_started:
            display = setup_frame.copy()
            if len(ignore_pts) >= 2:
                cv2.polylines(display, [np.array(ignore_pts)], True, (255, 0, 255), 2)
                for pt in ignore_pts:
                    cv2.circle(display, pt, 5, (255, 0, 255), -1)
            cv2.putText(display,
                        f"Draw ignore zone (L-click). R-click to clear. Press S to start.",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display,
                        f"thr_waiter={WAITER_THRESHOLD:.2f}  thr_uncertain={UNCERTAIN_THRESHOLD:.2f}  top_k={TOP_K_GALLERY}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # INITIALIZE VIDEO RECORDER HERE
                h, w = setup_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter('inference_output.mp4', fourcc, native_fps, (w, h))

                inference_started = True
                print("[Inference] Started. Recording to inference_output.mp4")
            elif key == ord('q'):
                break
            continue

        # ── Inference mode ────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.track(
            frame, persist=True, classes=[0],
            tracker="botsort.yaml", verbose=False
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            active_ids = set(ids.tolist())

            # Process locked IDs first (renders on top)
            sorted_indices = sorted(
                range(len(ids)),
                key=lambda k: ids[k] in stable_tracker.locked,
                reverse=True
            )
            active_boxes = []

            for idx in sorted_indices:
                box = boxes[idx]
                track_id = ids[idx]
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Skip if center is inside ignore zone
                if in_ignore_zone((cx, cy)):
                    continue

                # Skip duplicate detections (same center already processed)
                is_dup = any(
                    ax1 < cx < ax2 and ay1 < cy < ay2
                    for ax1, ay1, ax2, ay2 in active_boxes
                )
                if is_dup:
                    continue

                # ── Already locked ────────────────────────────────────────────
                final_info = stable_tracker.locked.get(track_id)
                if final_info:
                    name, color = final_info
                    display_text = f"ID:{track_id} {name} [LOCKED]"

                else:
                    # ── Run embedding every EMBED_EVERY_N frames ──────────────
                    current_label_info = None

                    if frame_count % EMBED_EVERY_N == 0:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        emb = get_embedding(model, crop, device)

                        if emb is not None:
                            dist = gallery_distance(emb, gallery, TOP_K_GALLERY)
                            label, col = classify(dist)
                            current_label_info = (label, col)
                            last_embeddings[track_id] = (label, col, dist)

                    # Use last known embedding if we skipped this frame
                    elif track_id in last_embeddings:
                        label, col, dist = last_embeddings[track_id]
                        current_label_info = (label, col)

                    # Update vote tracker
                    lock_status = stable_tracker.update(track_id, current_label_info)

                    if lock_status:
                        name, color = lock_status
                        display_text = f"ID:{track_id} {name} [LOCKED]"
                    elif track_id in stable_tracker.history and stable_tracker.history[track_id]:
                        name, color = stable_tracker.history[track_id][-1]
                        # Show live distance if available
                        if track_id in last_embeddings:
                            dist = last_embeddings[track_id][2]
                            display_text = f"ID:{track_id} {name} {dist:.2f}"
                        else:
                            display_text = f"ID:{track_id} {name}"
                    else:
                        name, color = "Analyzing...", (200, 200, 200)
                        display_text = f"ID:{track_id} {name}"

                active_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, display_text,
                            (x1, max(y1 - 10, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Cleanup stale track IDs every 60 frames
            if frame_count % 60 == 0:
                stable_tracker.cleanup(active_ids)

        # ── HUD overlay ───────────────────────────────────────────────────────
        cv2.putText(frame,
                    f"epoch={epoch}  thr={WAITER_THRESHOLD:.2f}/{UNCERTAIN_THRESHOLD:.2f}  top_k={TOP_K_GALLERY}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # WRITE FRAME TO VIDEO
        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow(win_name, frame)
        frame_count += 1

        # Precision sleep for 0.8x playback
        elapsed = time.perf_counter() - loop_start
        sleep_time = target_frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # CLEANUP VIDEO WRITER
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        checkpoint="/home/dior/Projects/Uniform3/checkpoints/best.pt",
        video_path="/home/dior/Projects/Uniform3/videos/perfetto.mp4",
        samples="/home/dior/Projects/Uniform3/samples/",
    )