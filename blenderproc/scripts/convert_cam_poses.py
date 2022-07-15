import argparse
from pathlib import Path
import pickle
import numpy as np


def convert_to_custom_format(annotation_root, output_path):
    with open(output_path, 'w') as out:
        list_of_annots = sorted([y for y in annotation_root.iterdir() if y.name.endswith('.pkl')])
        with open(list_of_annots[0], 'rb') as fptr:
            data = pickle.load(fptr)
            res_x = str(int(round(data['cam_K'][0][2])) * 2)
            res_y = str(int(round(data['cam_K'][1][2])) * 2)
            cam_k = [str(n) for n in np.array(data['cam_K']).ravel().tolist()]
            out.write(f'{res_x} {res_y} ' + ' '.join(cam_k) + '\n')
        for x in list_of_annots:
            with open(x, 'rb') as fptr:
                data = pickle.load(fptr)
                cam_pose = [str(n) for n in np.array(data['cam2world_matrix']).ravel().tolist()]
                out.write(' '.join(cam_pose) + '\n')


def cli():
    parser = argparse.ArgumentParser("Convert cameras saved in 'annotation' folder to a format that can be consumed by CustomCameraLoader (for debugging)")
    parser.add_argument('annotation_root', type=str, help='Path to annotation folder')
    parser.add_argument('--output_path', default=None, required=False, help="Determines where the data is going to be saved. Default: sampled_cameras.txt in annotation_root")
    args = parser.parse_args()

    args.annotation_root = Path(args.annotation_root)
    if args.output_path is None:
        args.output_path = args.annotation_root / "sampled_cameras.txt"
    else:
        args.output_path = Path(args.output_path)

    convert_to_custom_format(args.annotation_root, args.output_path)


if __name__ == "__main__":
    cli()
