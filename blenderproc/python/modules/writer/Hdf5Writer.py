from pathlib import Path

from blenderproc.python.writer.WriterUtility import WriterUtility

import os

import bpy
import h5py
import numpy as np
import json

from blenderproc.python.modules.main.GlobalStorage import GlobalStorage
from blenderproc.python.modules.writer.WriterInterface import WriterInterface
from blenderproc.python.utility.Utility import Utility
import imageio
import pickle

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


class Hdf5Writer(WriterInterface):
    """ For each key frame merges all registered output files into one hdf5 file.

    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - append_to_existing_output
          - If true, the names of the output hdf5 files will be chosen in a way such that there are no collisions
            with already existing hdf5 files in the output directory. Default: False
          - bool
        * - compression
          - The compression technique that should be used when storing data in a hdf5 file.
          - string
        * - delete_temporary_files_afterwards
          - True, if all temporary files should be deleted after merging. Default value: True.
          - bool
        * - stereo_separate_keys
          - If true, stereo images are saved as two separate images \*_0 and \*_1. Default: False
            (stereo images are combined into one np.array (2, ...)).
          - bool
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)
        self._append_to_existing_output = self.config.get_bool("append_to_existing_output", False)
        self._output_dir = self._determine_output_dir(False)

    def run(self):
        if self._avoid_output:
            print("Avoid output is on, no output produced!")
            return

        if self._append_to_existing_output:
            frame_offset = 0
            # Look for hdf5 file with highest index
            for path in os.listdir(self._output_dir):
                if path.endswith(".hdf5"):
                    index = path[:-len(".hdf5")]
                    if index.isdigit():
                        frame_offset = max(frame_offset, int(index) + 1)
        else:
            frame_offset = 0

        with open(self.config.get_string("json_path"), 'r') as front_json:
            front_anno = json.load(front_json)
            scene_name = front_anno['uid']

        if self.config.get_bool('export_blender', fallback=False):
            output_folder = f"{self._output_dir}/{scene_name}/mesh"
            Path(output_folder).mkdir(exist_ok=True, parents=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(Path(output_folder) / "scene.blend"))

        Path(f"{self._output_dir}/{scene_name}").mkdir(exist_ok=True, parents=True)

        # Go through all frames
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

            # Create output hdf5 file
            hdf5_path = os.path.join(self._output_dir, scene_name, str(frame + frame_offset) + ".hdf5")
            with h5py.File(hdf5_path, "w") as f:

                if not GlobalStorage.is_in_storage("output"):
                    print("No output was designed in prior models!")
                    return
                # Go through all the output types
                print("Merging data for frame " + str(frame) + " into " + hdf5_path)

                for output_type in GlobalStorage.get("output"):
                    # Build path (path attribute is format string)
                    file_path = output_type["path"]
                    if '%' in file_path:
                        file_path = file_path % frame

                    # Check if file exists
                    if not os.path.exists(file_path):
                        # If not try stereo suffixes
                        path_l, path_r = WriterUtility._get_stereo_path_pair(file_path)
                        if not os.path.exists(path_l) or not os.path.exists(path_r):
                            raise Exception("File not found: " + file_path)
                        else:
                            use_stereo = True
                    else:
                        use_stereo = False

                    if use_stereo:
                        path_l, path_r = WriterUtility._get_stereo_path_pair(file_path)

                        img_l, new_key, new_version = self._load_and_postprocess(path_l, output_type["key"],
                                                                                   output_type["version"])
                        img_r, new_key, new_version = self._load_and_postprocess(path_r, output_type["key"],
                                                                                   output_type["version"])

                        if self.config.get_bool("stereo_separate_keys", False):
                            WriterUtility._write_to_hdf_file(f, new_key + "_0", img_l)
                            WriterUtility._write_to_hdf_file(f, new_key + "_1", img_r)
                        else:
                            data = np.array([img_l, img_r])
                            WriterUtility._write_to_hdf_file(f, new_key, data)

                    else:
                        data, new_key, new_version = self._load_and_postprocess(file_path, output_type["key"],
                                                                                output_type["version"])

                        WriterUtility._write_to_hdf_file(f, new_key, data)

                    WriterUtility._write_to_hdf_file(f, new_key + "_version", np.string_([new_version]))

                blender_proc_version = Utility.get_current_version()
                if blender_proc_version:
                    WriterUtility._write_to_hdf_file(f, "blender_proc_version", np.string_(blender_proc_version))

            # convert h5py to mainer format
            if self.config.get_bool('convert_to_mainer', fallback=False):
                output_folder = f"{self._output_dir}/{scene_name}"
                frame_idx = int(os.path.basename(hdf5_path).split('.')[0])

                with h5py.File(hdf5_path, "r") as data:
                    print("{}:".format(hdf5_path))
                    keys = [key for key in data.keys()]
                    # create mainer-style folder structure
                    for key in ["rgb", "depth", "normal", "inst", "sem", "room", "annotation"]:
                        os.makedirs(f"{output_folder}/{key}", exist_ok=True)
                    # export all image based data
                    imageio.imwrite(f"{output_folder}/rgb/{frame_idx:05}.png", np.array(data['colors']))
                    imageio.imwrite(f"{output_folder}/depth/{frame_idx:05}.exr", np.array(data['depth']))
                    imageio.imwrite(f"{output_folder}/normal/{frame_idx:05}.exr", np.array(data['normals']))
                    imageio.imwrite(f"{output_folder}/inst/{frame_idx:05}.png", np.array(data['segmap'])[..., 0])
                    imageio.imwrite(f"{output_folder}/sem/{frame_idx:05}.png", np.array(data['segmap'])[..., 1])

                    segmap = eval(np.array(data["segcolormap"]))
                    room_names = [x['room_name'] for x in segmap]
                    room_labels = ["void"]+sorted([x['instanceid'] for x in front_anno['scene']['room']])
                    room_map =  {room_name: room_labels.index(room_name) for room_name in room_names}
                    inst2room_map = {int(x['idx']): x['room_name'] for x in eval(np.array(data["segcolormap"]))}
                    inst2room_id = {inst_idx: room_map[room_name]  for inst_idx, room_name in inst2room_map.items()}

                    inst  = np.array(data['segmap'])[..., 0]
                    room = np.zeros_like(inst)
                    for inst_idx, room_id in inst2room_id.items():
                        room[inst==inst_idx] = room_id
                    imageio.imwrite(f"{output_folder}/room/{frame_idx:05}.png", room)
                    # raw frame-wise annotation
                    frame_anno = {"segcolormap": eval(np.array(data["segcolormap"]))}
                    frame_anno.update(eval(np.array(data["campose"]))[0])

                    pickle.dump(frame_anno, open(f"{output_folder}/annotation/{frame_idx:05}.pkl", "wb"))

                os.remove(hdf5_path)
