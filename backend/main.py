from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import os
import uuid
import cv2
import shutil

from utils.pipeline import process_image

app = FastAPI()

BASE = "storage"

folders = [
    "classified/dent",
    "classified/hole",
    "classified/deframe",
    "unknown",
    "human_review",
    "no_damage"
]

for f in folders:
    os.makedirs(os.path.join(BASE, f), exist_ok=True)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    filename = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(BASE, filename)

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    status, label, confidence, img = process_image(temp_path)

    processed_path = os.path.join(BASE, "human_review", "processed_" + filename)
    cv2.imwrite(processed_path, img)

    os.remove(temp_path)

    if status == "classified" and label:
        class_folder = label.lower()
        final_path = os.path.join(BASE, "classified", class_folder, "processed_" + filename)
        shutil.move(processed_path, final_path)

        return {
            "status": "classified",
            "label": label,
            "confidence": confidence,
            "image_path": final_path
        }

    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "image_path": processed_path
    }


@app.post("/feedback")
async def feedback(
    image_path: str = Form(...),
    damage: str = Form(...),
    damage_type: str = Form(...)
):

    filename = os.path.basename(image_path)

    if damage == "no":
        final_path = os.path.join(BASE, "no_damage", filename)

    else:
        if damage_type == "unknown":
            final_path = os.path.join(BASE, "unknown", filename)
        else:
            final_path = os.path.join(BASE, "classified", damage_type.lower(), filename)

    shutil.move(image_path, final_path)

    return {"message": "Feedback saved"}