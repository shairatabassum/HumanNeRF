from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import numpy as np

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

folder_path = "../HumanNeRF_Dataset/images_v9/"
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
bbox_list = np.empty((0,4))

for IMAGE_PATH in image_files:
    #IMAGE_PATH = "frame_000012.png"
    IMAGE_PATH=os.path.join(folder_path,IMAGE_PATH)
    TEXT_PROMPT = "person"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame, xyxy = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    #cv2.imwrite("annotated_image.jpg", annotated_frame)
    bbox_list = np.vstack((bbox_list, xyxy))

np.savetxt('bboxes_v9.txt',bbox_list)
