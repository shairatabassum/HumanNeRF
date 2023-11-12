########################################################
# This script will read the output pkl file of VIBE
# Read pred_cam, bboxes, pose and betas
# Calculate intrinsic and extrinsic camera parameters
# Convert them to preapre metadata.json file
# Which will be suitable to train human-nerf model
########################################################

import joblib
import numpy as np
import json

output = joblib.load('output/V9.MOV/vibe_output.pkl')

#to check how many human objects are there in the video
print(f"Track ids for each subject appearaing in the video: {output.keys()}\n")

#to check the shape of each parameter
for k,v in output[1].items():
    if k!='joints2d':
        print(k, v.shape)

#taking the requied parameters
pred_cam = output[1]['pred_cam']
bboxes = output[1]['bboxes']
poses = output[1]['pose']
betas = output[1]['betas']

#number of frames in this video
total_frames = pred_cam.shape[0]
print(f"\nTotal frames: {total_frames}\n")



####################################################################
# Calculating camera parameteres from pred_cam and bboxes parameters
####################################################################
def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics
    
  
# calling the function, passing pred_cam and bboxes to get camera parameters
print('\n\nCalculating camera intrinsic and extrinsic parameters....')

cam_intrinsics = np.empty((total_frames, 3, 3), dtype=float)
cam_extrinsics = np.empty((total_frames, 4, 4), dtype=float)
for x in range(total_frames):
    cam_intrinsics[x], cam_extrinsics[x] = get_camera_parameters(pred_cam[x], bboxes[x])
    
print('\nFinished...!\n')



########################################################
#defining nested dictionary for generating metadata file
########################################################
print('\nGenerating metadata....\n')

#defining item ids as image filenames
item_ids = [f"frame_{i:06d}" for i in range(0, total_frames)]

metadata_dict = {}
#iterate through item_ids and populate the dictionary
i=0
for item_id in item_ids:
    item_data = {
        "poses": poses[i],
        "betas": betas[i],
        "cam_intrinsics": cam_intrinsics[i],
        "cam_extrinsics": cam_extrinsics[i]
    }
    metadata_dict[item_id] = item_data
    i=i+1


# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open("metadata/v9.json","w") as outfile:
    outfile.write(json.dumps(metadata_dict, cls=json_serialize, ensure_ascii=False))

print("Saved all information as 'metadata.json'\n")











