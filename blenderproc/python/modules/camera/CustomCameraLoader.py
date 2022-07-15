from pathlib import Path

import bpy

from blenderproc.python.modules.camera.CameraInterface import CameraInterface
from blenderproc.python.modules.utility.Config import Config
from blenderproc.python.modules.utility.ItemCollection import ItemCollection


class CustomCameraLoader(CameraInterface):

    def __init__(self, config):
        CameraInterface.__init__(self, config)
        # format used:
        # line 1: 2+8 intrinsic params (resx, resy, cam_k)
        # line 2: pose 1 extrinsics, 16 params
        # line 3: pose 2 extrinsics, 16 params
        # line 4: pose 3 extrinsics, 16 params
        # ...

    def run(self):
        # Set intrinsics
        filelines = Path(self.config.get_string("path")).read_text().splitlines()
        intrinsics = [float(x) for x in filelines[0].strip().split(' ')]
        intrinsics_config = Config({
            "resolution_x": int(intrinsics[0]),
            "resolution_y": int(intrinsics[1]),
            "cam_K": intrinsics[2:]
        })
        self._set_cam_intrinsics(bpy.context.scene.camera.data, intrinsics_config)
        # set extrinsics
        for pose_line in filelines[1:]:
            if pose_line.strip() != '':
                self._add_cam_pose(Config({
                    "cam2world_matrix": [float(x) for x in pose_line.strip().split(' ')]
                }))

    def _add_cam_pose(self, config):
        """ Adds new cam pose + intrinsics according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new cam pose.
        """

        # Collect camera object
        cam_ob = bpy.context.scene.camera

        # Set extrinsics from config
        self._set_cam_extrinsics(config)
