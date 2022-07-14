import blenderproc as bproc
import argparse
import os
from pathlib import Path
import bpy
import json

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

with open(args.front, 'r') as front_json:
    front_anno = json.load(front_json)
    scene_name = front_anno['uid']

dest_path = Path(args.output_dir, scene_name, "mesh")
dest_path.mkdir(exist_ok=True, parents=True)

bpy.ops.export_scene.obj(filepath=str(dest_path / "mesh.obj"), use_materials=False)
