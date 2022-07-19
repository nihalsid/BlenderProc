import argparse
import os
from cv2 import dft

import h5py
import numpy as np
import imageio
import pickle
import glob
import json

def remap_labeled_img(lab_img, mapping, default=-1):
    new_lab_img = np.full(lab_img.shape,default, dtype=lab_img.dtype)
    for label in np.unique(lab_img):
        label_mask = lab_img == label
        new_lab_img[label_mask] = mapping[label]
    return new_lab_img

def convert_hdf5(base_file_path, front_anno, running_unique_uid2asset_id, output_dir=None):
    if os.path.exists(base_file_path):
        if os.path.isfile(base_file_path):
            scene_name = front_anno['uid']
            output_folder = f"{output_dir}/{scene_name}"
            frame_idx = int(os.path.basename(base_file_path).split('.')[0])
            with h5py.File(base_file_path, 'r') as data:
                print("{}:".format(base_file_path))
                keys = [key for key in data.keys()]
                # create mainer-style folder structure
                for key in ["rgb", "depth", "normal", "inst", "sem", "room", "asset", "annotation"]:
                    os.makedirs(f"{output_folder}/{key}", exist_ok=True)
                # export all image based data
                imageio.imwrite(
                    f"{output_folder}/rgb/{frame_idx:05}.png", np.array(data['colors']))
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
                # get asset ids for instances, update a global mapping
                
                idx2unique_uid = dict()
                for seg_entry in segmap:
                    idx = int(seg_entry['idx'])
                    unique_uid = seg_entry['unique_uid']
                    if unique_uid not in running_unique_uid2asset_id:
                        asset_id = len(running_unique_uid2asset_id)
                        running_unique_uid2asset_id[unique_uid] = asset_id
                    if idx in idx2unique_uid and idx2unique_uid[idx] != unique_uid:
                        # sanity check
                        print(f"ERROR: {idx}")
                    idx2unique_uid[idx] = unique_uid
                # map idx -> asset_id
                idx2asset_id = {idx: running_unique_uid2asset_id[unique_uid] for  idx, unique_uid in idx2unique_uid.items()}
                # also store asset mask
                asset = remap_labeled_img(inst, idx2asset_id)
                imageio.imwrite(
                    f"{output_folder}/asset/{frame_idx:05}.png", asset)
                # raw frame-wise annotation
                frame_anno = {"segcolormap": segmap}
                frame_anno.update(eval(np.array(data["campose"]))[0])
                frame_anno.update({"room_map": room_map, "inst2room_id": inst2room_id})
                # mappings for instance/idx to asset_id
                frame_anno.update({
                    "idx2asset_id": idx2asset_id,
                    "unique_uid2asset_id": running_unique_uid2asset_id, 
                    "idx2unique_uid": idx2unique_uid})


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
    # assigns unique_uid -> asset_id by updating mapping over frames
    running_unique_uid2asset_id = {} 

    if args.hdf5.endswith(".hdf5"):
        # single file
        convert_hdf5(args.hdf5, args.output_dir)
    else:
        scene_name = str(os.path.basename(args.hdf5)).split('.')[-1]
        front_anno = json.load(open(f"{args.front3d_root}/{scene_name}.json", 'r'))
        for hdf5_fn in glob.glob(f"{args.hdf5}/*.hdf5"):
            convert_hdf5(hdf5_fn, front_anno, running_unique_uid2asset_id, args.output_dir)


if __name__ == "__main__":
    cli()
