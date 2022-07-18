import numpy as np
import pickle
from dvis import dvis

def trs2f_vec(trs):
    # get nx6 forward vector as vec_origin, vec_target
    return np.concatenate([trs[:,:3,3],trs[:,:3,3]+ trs[:,:3,2]],1)

scene_name ="/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1" 
compl_trajectory = pickle.load(open(f"{scene_name}/compl_trajectory_2d.pkl", 'rb'))
dvis(trs2f_vec(compl_trajectory),'vec',name='vec',vs=0.1,c=-1)
[dvis(compl_trajectory[i],t=i,vs=0.5, name=f"trs/{i}") for i in range(100)]