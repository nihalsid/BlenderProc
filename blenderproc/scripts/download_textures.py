import json
import os
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm
import shutil


def download_texture_for_url(given_url, front_3D_texture_path):
    hash_nr = given_url.split("/")[-2]
    hash_folder = os.path.join(front_3D_texture_path, hash_nr)
    try:
        if not os.path.exists(hash_folder):
            # download the file
            os.makedirs(hash_folder)
            print(f"This texture: {hash_nr} could not be found it will be downloaded.")
            # replace https with http as ssl connection out of blender are difficult
            urlretrieve(given_url, os.path.join(hash_folder, "texture.png"))
            if not os.path.exists(os.path.join(hash_folder, "texture.png")):
                raise Exception(f"The texture could not be found, the following url was used: "
                                f"{front_3D_texture_path}, this is the extracted hash: {hash_nr}, "
                                f"given url: {given_url}")
    except Exception as err:
        print(err)
        if os.path.exists(hash_folder):
            shutil.rmtree(hash_folder)
    return hash_folder


def download_textures(root_json, root_textures, nproc, pid):
    all_jsons = sorted([x for x in root_json.iterdir() if x.name.endswith('.json')])
    all_jsons = [x for i, x in enumerate(all_jsons) if i % nproc == pid]
    for jsonfile in tqdm(all_jsons, colour='green'):
        data = json.loads(Path(jsonfile).read_text())
        materials_needing_textures = [x for x in data['material'] if x['texture'] != '']
        for mat in materials_needing_textures:
            download_texture_for_url(mat['texture'], root_textures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Download all 3D Front Textures")
    parser.add_argument('front3d_root', type=str, help='Path to 3dfront json folder')
    parser.add_argument('texture_root', type=str, help='Path to exported textures')
    parser.add_argument('nproc', type=int, help='total procs')
    parser.add_argument('pid', type=int, help='proc id')
    args = parser.parse_args()
    download_textures(Path(args.front3d_root), Path(args.texture_root), args.nproc, args.pid)
