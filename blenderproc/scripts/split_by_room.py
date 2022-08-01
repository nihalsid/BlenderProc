import argparse
import shutil
from pathlib import Path
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import imageio
import pickle


def cli_old_format():
    parser = argparse.ArgumentParser("Convert cameras saved in 'annotation' folder to a format that can be consumed by CustomCameraLoader (for debugging)")
    parser.add_argument('front3d_root', type=str, help='Path to 3dfront json folder')
    parser.add_argument('scene_path', type=str, help='Path to exported scene folder')
    parser.add_argument('--output_path', default=None, required=False, help="Determines where the data is going to be saved. Default: in scene_path as a subfolder called room_splits")
    args = parser.parse_args()

    args.scene_path = Path(args.scene_path)
    if args.output_path is None:
        args.output_path = args.scene_path / "room_splits"
    else:
        args.output_path = Path(args.output_path)
    args.output_path.mkdir(exist_ok=True)

    front_anno = json.load(open(f"{args.front3d_root}/{args.scene_path.stem}.json", 'r'))
    room_labels = ["void"] + sorted([x['instanceid'] for x in front_anno['scene']['room']])
    FOLDER_LIST = ["rgb", "depth", "normal", "inst", "sem", "room", "asset", "annotation"]
    EXT = ["png", "exr", "exr", "png", "png", "png", "png", "pkl"]

    for room in room_labels:
        (args.output_path / room).mkdir(exist_ok=True)
        for folder in FOLDER_LIST:
            (args.output_path / room / folder).mkdir(exist_ok=True)

    for label_room_path in tqdm(list((args.scene_path / "room").iterdir())):
        label_room = np.array(Image.open(label_room_path))
        unique_rooms = np.unique(label_room)
        for unique_room in unique_rooms:
            current_room = room_labels[unique_room]
            for folder, ext in zip(FOLDER_LIST, EXT):
                shutil.copyfile(args.scene_path / folder / f'{label_room_path.stem}.{ext}', args.output_path / current_room / folder / f'{label_room_path.stem}.{ext}')
            room_mask = label_room == unique_room
            rgb_mask = np.array(Image.open(args.scene_path / "rgb" / f'{label_room_path.stem}.{EXT[0]}'))
            rgb_mask[np.logical_not(room_mask), :] = 0
            (args.output_path / current_room / "room_mask").mkdir(exist_ok=True)
            (args.output_path / current_room / "rgb_mask").mkdir(exist_ok=True)
            (args.output_path / current_room / "depth_npz").mkdir(exist_ok=True)
            Image.fromarray(room_mask).save(args.output_path / current_room / "room_mask" / f'{label_room_path.stem}.png')
            Image.fromarray(rgb_mask).save(args.output_path / current_room / "rgb_mask" / f'{label_room_path.stem}.{EXT[0]}')
            depth_arr = np.array(imageio.v2.imread(args.output_path / current_room / "depth" / f'{label_room_path.stem}.exr'), dtype=np.float32)
            np.savez_compressed(str(args.output_path / current_room / "depth_npz" / f'{label_room_path.stem}.npz'), arr=depth_arr)


def cli_new_format():
    parser = argparse.ArgumentParser("Convert cameras saved in 'annotation' folder to a format that can be consumed by CustomCameraLoader (for debugging)")
    parser.add_argument('front3d_root', type=str, help='Path to 3dfront json folder')
    parser.add_argument('scene_path', type=str, help='Path to exported scene folder')
    parser.add_argument('--output_path', default=None, required=False, help="Determines where the data is going to be saved. Default: in scene_path as a subfolder called room_splits")
    args = parser.parse_args()

    args.scene_path = Path(args.scene_path)
    if args.output_path is None:
        args.output_path = args.scene_path / "room_splits"
    else:
        args.output_path = Path(args.output_path)
    args.output_path.mkdir(exist_ok=True)

    front_anno = json.load(open(f"{args.front3d_root}/{args.scene_path.stem}.json", 'r'))
    room_labels = ["void"] + sorted([x['instanceid'] for x in front_anno['scene']['room']])
    FOLDER_LIST = ["rgb", "depth", "normal", "inst", "sem", "room", "asset", "annotation"]
    EXT = ["png", "exr", "exr", "png", "png", "png", "png", "pkl"]

    data_split_fn = args.scene_path / "compl_trajectory_split.pkl"
    data_room_fn = args.scene_path / "compl_trajectory_room.pkl"
    split_indices = pickle.load(open(data_split_fn,'rb'))
    room_ids = pickle.load(open(data_room_fn,'rb'))

    for room in room_labels:
        (args.output_path / room).mkdir(exist_ok=True)
        for folder in FOLDER_LIST:
            (args.output_path / room / folder).mkdir(exist_ok=True)

    for target_room_id in tqdm(np.unique(room_ids)):
        target_frame_inds = np.argwhere(room_ids == target_room_id)

        file_stem = f'{target_frame_inds[0][0]:05}'
        unique_room_labels, unique_room_label_ctr = np.unique(np.array(Image.open(args.scene_path / "room" / f'{file_stem}.png')), return_counts=True)
        target_room_id = unique_room_labels[unique_room_label_ctr.argmax()]

        current_room = room_labels[target_room_id]
        room_rgb_train = []
        room_rgb_val = []
        for frame_idx in target_frame_inds:
            frame_idx = int(frame_idx)
            file_stem = f'{frame_idx:05}'
            if split_indices[frame_idx]:
                room_rgb_val.append(frame_idx)
            else:
                room_rgb_train.append(frame_idx)
            for folder, ext in zip(FOLDER_LIST, EXT):
                shutil.copyfile(args.scene_path / folder / f'{file_stem}.{ext}', args.output_path / current_room / folder / f'{file_stem}.{ext}')
            label_room = np.array(Image.open(args.scene_path / "room" / f'{file_stem}.png'))

            room_mask = label_room == target_room_id
            rgb_mask = np.array(Image.open(args.scene_path / "rgb" / f'{file_stem}.{EXT[0]}'))
            rgb_mask[np.logical_not(room_mask), :] = 0
            (args.output_path / current_room / "room_mask").mkdir(exist_ok=True)
            (args.output_path / current_room / "rgb_mask").mkdir(exist_ok=True)
            (args.output_path / current_room / "depth_npz").mkdir(exist_ok=True)
            Image.fromarray(room_mask).save(args.output_path / current_room / "room_mask" / f'{file_stem}.png')
            Image.fromarray(rgb_mask).save(args.output_path / current_room / "rgb_mask" / f'{file_stem}.{EXT[0]}')
            depth_arr = np.array(imageio.v2.imread(args.output_path / current_room / "depth" / f'{file_stem}.exr'), dtype=np.float32)
            np.savez_compressed(str(args.output_path / current_room / "depth_npz" / f'{file_stem}.npz'), arr=depth_arr)
            split = {
                'train': room_rgb_train,
                'val': room_rgb_val,
            }
            Path(str(args.output_path / current_room / "split.json")).write_text(json.dumps(split))

if __name__ == "__main__":
    cli_new_format()
