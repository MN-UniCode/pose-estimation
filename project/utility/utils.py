import numpy as np

# Lista di 33 landmark MediaPipe
# Riempiti con zeri o None se mancanti
def yolo_to_mediapipe(kpts):
    mp33 = np.zeros((33, 3), dtype=float)

    coco2mp = {
        0: 0,    # nose
        5: 11,   # L shoulder
        7: 13,   # L elbow
        9: 15,   # L wrist
        6: 12,   # R shoulder
        8: 14,   # R elbow
        10: 16,  # R wrist
        11: 23,  # L hip
        13: 25,  # L knee
        15: 27,  # L ankle
        12: 24,  # R hip
        14: 26,  # R knee
        16: 28,  # R ankle
    }

    for coco_idx, mp_idx in coco2mp.items():
        mp33[mp_idx] = kpts[coco_idx]

    return mp33
