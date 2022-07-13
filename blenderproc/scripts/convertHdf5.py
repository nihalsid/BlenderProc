import argparse
import os
from cv2 import dft

import h5py
import numpy as np
import imageio
import pickle
import glob

def convert_hdf5(base_file_path, output_folder=None):
    if os.path.exists(base_file_path):
        if os.path.isfile(base_file_path):
            base_name = str(os.path.basename(base_file_path)).split('.')[0]
            frame_idx = int(os.path.basename(base_file_path).split('.')[0])
            with h5py.File(base_file_path, 'r') as data:
                print("{}:".format(base_file_path))
                keys = [key for key in data.keys()]
                # create mainer-style folder structure
                for key in ["rgb", "depth", "normal", "inst", "sem", "annotation"]:
                    os.makedirs(f"{output_folder}/{key}", exist_ok=True)
                # export all image based data
                imageio.imwrite(
                    f"{output_folder}/rgb/{frame_idx:05}.jpg", np.array(data['colors']))
                imageio.imwrite(
                    f"{output_folder}/depth/{frame_idx:05}.exr", np.array(data['depth']))
                imageio.imwrite(
                    f"{output_folder}/normal/{frame_idx:05}.exr", np.array(data['normals']))
                imageio.imwrite(
                    f"{output_folder}/inst/{frame_idx:05}.png", np.array(data['segmap'])[..., 0])
                imageio.imwrite(
                    f"{output_folder}/sem/{frame_idx:05}.png", np.array(data['segmap'])[..., 1])

                # raw frame-wise annotation
                frame_anno = {"segcolormap": eval(
                    np.array(data["segcolormap"]))}
                frame_anno.update(eval(np.array(data["campose"]))[0])

                pickle.dump(frame_anno, open(
                    f"{output_folder}/annotation/{frame_idx:05}.pkl", "wb"))
        else:
            print("The path is not a file")
    else:
        print("The file does not exist: {}".format(base_file_path))


def cli():
    parser = argparse.ArgumentParser(
        "Script to save images out of (a) hdf5 file(s).")
    parser.add_argument('hdf5', type=str, help='Path to hdf5 file/s')
    parser.add_argument('--output_dir', default=None,
                        help="Determines where the data is going to be saved. Default: Current directory")

    args = parser.parse_args()
    
    if args.hdf5.endswith(".hdf5"):
        # single file
        convert_hdf5(args.hdf5, args.output_dir)
    else:
        for hdf5_fn in glob.glob(f"{args.hdf5}/*.hdf5"):
            convert_hdf5(hdf5_fn, args.output_dir)


if __name__ == "__main__":
    cli()
