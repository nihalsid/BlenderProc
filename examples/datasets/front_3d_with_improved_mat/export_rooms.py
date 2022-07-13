import blenderproc as bproc
import argparse
import os
from pathlib import Path
import bpy

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

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

room_split_floor_objects = bproc.loader.load_front3d_floors_split_by_room(json_path=args.front, future_model_path=args.future_folder)

context = bpy.context
scene = context.scene
viewlayer = context.view_layer

dest_path = Path(args.output_dir, "room_floor")
dest_path.mkdir(exist_ok=True)

for room_name in room_split_floor_objects:
    bpy.ops.object.select_all(action='DESELECT')
    obs = [o.blender_obj for o in room_split_floor_objects[room_name]]
    for ob in obs:
        viewlayer.objects.active = ob
        ob.select_set(True)
    stl_path = dest_path / f"{room_name}.stl"
    bpy.ops.export_mesh.stl(filepath=str(stl_path), use_selection=True)

bpy.ops.object.select_all(action='DESELECT')
