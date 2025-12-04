# detector.py
import os
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.cluster import DBSCAN
from PIL import Image


MODEL_NAME = "yolov8n.pt"  


ANIMAL_CLASSES = {"cow", "sheep", "horse", "dog", "cat"}  


_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model

def detect_animals(image_path, conf_thresh=0.3):
    """
    Runs YOLO on the image and returns detections filtered to animal classes.
    Returns: list of detections: dict {xmin,ymin,xmax,ymax,conf,label}
    """
    model = get_model()
    
    results = model.predict(source=image_path, conf=conf_thresh, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes  
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy())
            cls_name = model.model.names[cls_id] if hasattr(model, "model") else str(cls_id)
            if cls_name in ANIMAL_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "xmin": int(x1), "ymin": int(y1),
                    "xmax": int(x2), "ymax": int(y2),
                    "conf": conf, "label": cls_name
                })
    return detections

def cluster_herds(detections, eps_px=100, min_samples=3):
    """
    Cluster detections based on centroid proximity (DBSCAN)
    eps_px: neighborhood radius in pixels
    min_samples: minimum animals in a cluster to be considered a herd
    Returns clusters: list of dict {members:[idxs], centroid:(x,y)}
    """
    if len(detections) == 0:
        return []

    centroids = []
    for d in detections:
        cx = (d["xmin"] + d["xmax"]) / 2.0
        cy = (d["ymin"] + d["ymax"]) / 2.0
        centroids.append([cx, cy])
    X = np.array(centroids)

    db = DBSCAN(eps=eps_px, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        idxs = np.where(labels == label)[0].tolist()
        if len(idxs) >= min_samples:
            pts = X[idxs]
            centroid = (float(pts[:,0].mean()), float(pts[:,1].mean()))
            clusters.append({"label": int(label), "members": idxs, "centroid": centroid})
    return clusters

def annotate_image(image_path, detections, clusters, out_path):
    """
    Draw boxes and cluster info on image and save to out_path.
    """
    img = cv2.imread(image_path)
    
    for i, d in enumerate(detections):
        color = (0, 255, 0)
        cv2.rectangle(img, (d["xmin"], d["ymin"]), (d["xmax"], d["ymax"]), color, 2)
        cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (d["xmin"], d["ymin"]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    
    colors = [
        (0,0,255), (255,0,0), (0,255,255), (255,0,255),
        (125,125,0), (0,125,125)
    ]
    for cluster in clusters:
        idxs = cluster["members"]
        c = colors[cluster["label"] % len(colors)]
        
        pts = []
        for idx in idxs:
            d = detections[idx]
            pts.append([d["xmin"], d["ymin"]])
            pts.append([d["xmax"], d["ymax"]])
        pts = np.array(pts)
        x1, y1 = int(pts[:,0].min()), int(pts[:,1].min())
        x2, y2 = int(pts[:,0].max()), int(pts[:,1].max())
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 3)
        cx, cy = int(cluster["centroid"][0]), int(cluster["centroid"][1])
        cv2.putText(img, f'Herd ({len(idxs)})', (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    cv2.imwrite(out_path, img)
    return out_path
