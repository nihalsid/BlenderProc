from src.utility.SetupUtility import SetupUtility
SetupUtility.setup([])

from src.utility.WriterUtility import WriterUtility
from src.utility.Initializer import Initializer
from src.utility.loader.ObjectLoader import ObjectLoader
from src.utility.CameraUtility import CameraUtility
from src.utility.LightUtility import Light
from mathutils import Matrix, Vector, Euler

from src.utility.RendererUtility import RendererUtility
from src.utility.PostProcessingUtility import PostProcessingUtility

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('camera', help="Path to the camera file, should be examples/basic/camera_positions")
parser.add_argument('scene', help="Path to the scene.obj file, should be examples/basic/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basic/output")
args = parser.parse_args()

Initializer.init()

objs = ObjectLoader.load(args.scene)

light = Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

CameraUtility.set_intrinsics_from_blender_params(1, 512, 512, lens_unit="FOV")
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        matrix_world = Matrix.Translation(Vector(line[:3])) @ Euler(line[3:6], 'XYZ').to_matrix().to_4x4()
        CameraUtility.add_camera_pose(matrix_world)

RendererUtility.enable_distance_output()
RendererUtility.enable_normals_output()
RendererUtility.set_samples(20)
RendererUtility.toggle_stereo(False)
data = RendererUtility.render()

data["distance"] = PostProcessingUtility.trim_redundant_channels(data["distance"])
data['depth'] = PostProcessingUtility.dist2depth(data['distance'])

WriterUtility.save_to_hdf5(args.output_dir, data)
