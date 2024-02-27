import numpy as np
import pickle
import os
import skimage.io as io

image_dir = './assets/P800853_11_10_22_run3-ezgifcom-video-cutter_Instrument'

segm_mask = [
    [
        i, np.array(io.imread(os.path.join(image_dir, image)))
    ] for i, image in enumerate(os.listdir(image_dir))
]

with open('{}.pkl'.format(image_dir.split('/')[-1]), 'wb') as pkl:
    pickle.dump(segm_mask, pkl)
