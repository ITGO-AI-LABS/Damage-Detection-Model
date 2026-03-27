from ultralytics import YOLO
import cv2

det_model = YOLO("models/segmentation_weights/best.pt")
cls_model = YOLO("models/classification_weights/best.pt")


def process_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not loaded")

    status = "human_review"
    label = None
    confidence = 0

    results = det_model.predict(image_path, conf=0.05, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:

            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            cls_result = cls_model.predict(crop, verbose=False)

            if len(cls_result[0].boxes) > 0:
                cls_id = int(cls_result[0].boxes.cls[0])
                cls_conf = float(cls_result[0].boxes.conf[0])
                label = cls_model.names[cls_id]

                if cls_conf >= 0.4:
                    status = "classified"
                elif cls_conf >= 0.2:
                    status = "unknown"
                else:
                    status = "human_review"

                confidence = cls_conf

                text = f"{label} ({cls_conf:.2f})"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return status, label, confidence, img