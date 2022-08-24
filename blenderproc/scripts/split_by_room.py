import argparse
import shutil
from pathlib import Path
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import imageio
import pickle


def process_room(front3d_root, scene_path, output_path):
    VALID_ROOMS = ['SecondBedroom', 'ElderlyRoom', 'BedRoom', 'Bedroom', 'MasterBedroom', 'LivingDiningRoom', 'NannyRoom', 'KidsRoom', 'DiningRoom', 'LivingRoom']

    front_anno = json.load(open(f"{front3d_root}/{scene_path.stem}.json", 'r'))
    room_labels = ["void"] + sorted([x['instanceid'] for x in front_anno['scene']['room']])
    FOLDER_LIST = ["rgb", "depth", "normal", "inst", "sem", "room", "asset", "annotation"]
    EXT = ["png", "exr", "exr", "png", "png", "png", "png", "pkl"]

    room_ids = pickle.load(open(scene_path / "render_split" / "room_ids.pkl", 'rb'))

    valid_room_labels = [x for x in room_labels if x.split('-')[0] in VALID_ROOMS]
    valid_unique_room_ids = list(set([x for x in room_ids if room_labels[x].split('-')[0] in VALID_ROOMS]))

    for room in valid_room_labels:
        (output_path / f"{scene_path.stem}_{room}").mkdir(exist_ok=True)
        for folder in FOLDER_LIST:
            (output_path / f"{scene_path.stem}_{room}" / folder).mkdir(exist_ok=True)

    for target_room_id in tqdm(valid_unique_room_ids, desc='room', colour='green'):
        target_frame_inds = np.argwhere(room_ids == target_room_id)
        current_room = room_labels[target_room_id]

        for frame_idx in target_frame_inds:
            frame_idx = int(frame_idx)
            file_stem = f'{frame_idx:05}'
            for folder, ext in zip(FOLDER_LIST, EXT):
                shutil.copyfile(scene_path / folder / f'{file_stem}.{ext}', output_path / f"{scene_path.stem}_{current_room}" / folder / f'{file_stem}.{ext}')
            label_room = np.array(Image.open(scene_path / "room" / f'{file_stem}.png'))
            room_mask = label_room == target_room_id
            rgb_mask = np.array(Image.open(scene_path / "rgb" / f'{file_stem}.{EXT[0]}'))
            rgb_mask[np.logical_not(room_mask), :] = 0
            (output_path / f"{scene_path.stem}_{current_room}" / "room_mask").mkdir(exist_ok=True)
            (output_path / f"{scene_path.stem}_{current_room}" / "rgb_mask").mkdir(exist_ok=True)
            (output_path / f"{scene_path.stem}_{current_room}" / "depth_npz").mkdir(exist_ok=True)
            Image.fromarray(room_mask).save(output_path / f"{scene_path.stem}_{current_room}" / "room_mask" / f'{file_stem}.png')
            Image.fromarray(rgb_mask).save(output_path / f"{scene_path.stem}_{current_room}" / "rgb_mask" / f'{file_stem}.{EXT[0]}')
            depth_arr = np.array(imageio.v2.imread(output_path / f"{scene_path.stem}_{current_room}" / "depth" / f'{file_stem}.exr'), dtype=np.float32)
            np.savez_compressed(str(output_path / f"{scene_path.stem}_{current_room}" / "depth_npz" / f'{file_stem}.npz'), arr=depth_arr)

        room_rgb_train = [int(x) for x in Path(scene_path / "room_split" / f"{target_room_id}_train.txt").read_text().splitlines()]
        room_rgb_val = [int(x) for x in Path(scene_path / "room_split" / f"{target_room_id}_val.txt").read_text().splitlines()]
        split = {
            'train': room_rgb_train,
            'val': room_rgb_val,
        }
        Path(str(output_path / f"{scene_path.stem}_{current_room}" / "split.json")).write_text(json.dumps(split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert cameras saved in 'annotation' folder to a format that can be consumed by CustomCameraLoader (for debugging)")
    parser.add_argument('front3d_root', type=str, help='Path to 3dfront json folder')
    parser.add_argument('scene_path', type=str, help='Path to exported scene folder')
    parser.add_argument('output_path', type=str, help="Determines where the data is going to be saved. Default: in scene_path as a subfolder called split_by_room")
    args = parser.parse_args()

    args.scene_path = Path(args.scene_path)
    args.output_path = Path(args.output_path)
    args.output_path.mkdir(exist_ok=True)

    all_scenes = list(Path(args.scene_path).iterdir())

    errored_out_rooms = []
    for scene in tqdm(all_scenes, colour='blue', desc='scene'):
        try:
            process_room(args.front3d_root, scene, args.output_path)
        except Exception as err:
            print('ERROR: ', err, 'for room', scene)
            errored_out_rooms.append((args.front3d_root, str(scene), str(args.output_path)))

    if len(errored_out_rooms) != 0:
        Path("err.txt").write_text("\n".join([','.join(x) for x in errored_out_rooms]))
