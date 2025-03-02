from pathlib import Path

import bpy
import pickle
import numpy as np

from blenderproc.python.modules.camera.CameraInterface import CameraInterface
from blenderproc.python.modules.utility.Config import Config
from blenderproc.python.modules.utility.ItemCollection import ItemCollection


class CustomCameraLoader2(CameraInterface):

    def __init__(self, config):
        CameraInterface.__init__(self, config)

    def run(self):
        camera_trajetory = pickle.load(open(self.config.get_string("path"),'rb'))
        max_n = self.config.get_int("max_n")
        if max_n<=0:
            max_n = len(camera_trajetory)
        stride = self.config.get_int("stride")
        if stride<=0:
            stride = 1

        ## TODO: HARDCODED FOR TESTING
        intrinsics_config = Config({
            "resolution_x": 256,
            "resolution_y": 256,
            "fov": 89
        })
        self._set_cam_intrinsics(bpy.context.scene.camera.data, intrinsics_config)
        
        # set extrinsics
        z_flip = np.diag([1,1,-1,1])
        for cam_trs in camera_trajetory[:max_n][::stride]:
            self._add_cam_pose(Config({
                    "cam2world_matrix": (cam_trs @ z_flip).tolist()
                }))

    def _add_cam_pose(self, config):
        """ Adds new cam pose + intrinsics according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new cam pose.
        """

        # Collect camera object
        cam_ob = bpy.context.scene.camera

        # Set extrinsics from config
        self._set_cam_extrinsics(config)
