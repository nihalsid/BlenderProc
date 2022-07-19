import blenderproc as bproc
import argparse
import os
from pathlib import Path
import bpy
import json

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the cam2world4D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
# export mesh
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

# export floor
room_split_floor_objects = bproc.loader.load_front3d_floors_split_by_room(json_path=args.front, future_model_path=args.future_folder)

context = bpy.context
scene = context.scene
viewlayer = context.view_layer

with open(args.front, 'r') as front_json:
    front_anno = json.load(front_json)
    scene_name = front_anno['uid']

dest_path = Path(args.output_dir, scene_name, "floor")
dest_path.mkdir(exist_ok=True, parents=True)

for room_name in room_split_floor_objects:
    bpy.ops.object.select_all(action='DESELECT')
    obs = [o.blender_obj for o in room_split_floor_objects[room_name]]
    for ob in obs:
        viewlayer.objects.active = ob
        ob.select_set(True)
    stl_path = dest_path / f"{room_name}.stl"
    bpy.ops.export_mesh.stl(filepath=str(stl_path), use_selection=True)

bpy.ops.object.select_all(action='DESELECT')