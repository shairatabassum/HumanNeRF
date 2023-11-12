import matplotlib.pyplot as plt
import cv2  # type: ignore
import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List

image = cv2.imread('im.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

print(masks)
