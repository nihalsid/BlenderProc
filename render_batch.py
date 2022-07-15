import subprocess
import glob
import os

front3d_root = os.path.expanduser("~/fb_data/front3d/3D-FRONT")
future3d_root = os.path.expanduser("~/fb_data/front3d/3D-FUTURE-model")
texture_root = os.path.expanduser("~/fb_data/front3d/3D-FRONT-texture")

num_scenes = 100
out_dir = os.path.expanduser("~/fb_data/renders_front3d/")
for scene_idx, scene_path in enumerate(glob.glob(f"{front3d_root}/*.json")):
    scene_name = scene_path.split('/')[-1].split('.')[0]  # TODO: hacky
    subprocess.Popen(["blenderproc", "run", "examples/datasets/front_3d_with_improved_mat/config.yaml",
                     scene_path, future3d_root, texture_root, "resources/cc_textures", f"{out_dir}/{scene_name}"]).wait()
    print(f"=== FINALIZED {scene_name}")
