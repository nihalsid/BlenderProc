import argparse
import os
from cv2 import dft

import h5py
import numpy as np
import imageio
import pickle
import glob
import json

def convert_hdf5(base_file_path, front_anno, output_dir=None):
    if os.path.exists(base_file_path):
        if os.path.isfile(base_file_path):
            scene_name = front_anno['uid']
            output_folder = f"{output_dir}/{scene_name}"
            frame_idx = int(os.path.basename(base_file_path).split('.')[0])
            with h5py.File(base_file_path, 'r') as data:
                print("{}:".format(base_file_path))
                keys = [key for key in data.keys()]
                # create mainer-style folder structure
                for key in ["rgb", "depth", "normal", "inst", "sem", "room", "annotation"]:
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

                segmap = eval(np.array(data["segcolormap"])) 
                room_names = [x['room_name'] for x in segmap]
                room_labels = ["void"]+sorted([x['instanceid'] for x in front_anno['scene']['room']])
                room_map =  {room_name: room_labels.index(room_name) for room_name in room_names}
                inst2room_name = {int(x['idx']): x['room_name'] for x in eval(np.array(data["segcolormap"]))}
                inst2room_id = {inst_idx: room_map[room_name]  for inst_idx, room_name in inst2room_name.items()}

                inst  = np.array(data['segmap'])[..., 0]
                room = np.zeros_like(inst)
                for inst_idx, room_id in inst2room_id.items():
                    room[inst==inst_idx] = room_id
                imageio.imwrite(
                    f"{output_folder}/room/{frame_idx:05}.png", room)
                # raw frame-wise annotation
                frame_anno = {"segcolormap": eval(
                    np.array(data["segcolormap"]))}
                frame_anno.update(eval(np.array(data["campose"]))[0])
                frame_anno.update({"room_map": room_map, "inst2room_id": inst2room_id})

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
    parser.add_argument('front3d_root', type=str, help='Path to FRONT3D')
    parser.add_argument('--output_dir', default=None,
                        help="Determines where the data is going to be saved. Default: Current directory")

    args = parser.parse_args()
    
    if args.hdf5.endswith(".hdf5"):
        # single file
        convert_hdf5(args.hdf5, args.output_dir)
    else:
        scene_name = str(os.path.basename(args.hdf5)).split('.')[-1]
        front_anno = json.load(open(f"{args.front3d_root}/{scene_name}.json", 'r'))
        for hdf5_fn in glob.glob(f"{args.hdf5}/*.hdf5"):
            convert_hdf5(hdf5_fn, front_anno, args.output_dir)


if __name__ == "__main__":
    cli()
