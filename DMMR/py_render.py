import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

scene = pyrender.Scene(bg_color=np.array([0.9,0.9,0.9]), ambient_light=[0.5, 0.5, 0.5, 1.0])

floor = trimesh.load('floor.obj')
#child = trimesh.load('child.obj')
#mom = trimesh.load('mom.obj')

floor_mesh = pyrender.Mesh.from_trimesh(floor, smooth=False)
scene.add(floor_mesh)
#child_mesh = pyrender.Mesh.from_trimesh(child)
#scene.add(child_mesh)
#par_mesh = pyrender.Mesh.from_trimesh(mom)
#scene.add(par_mesh)



pyrender.Viewer(scene, use_raymond_lighting=True)

#light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
#                          innerConeAngle=np.pi/16.0,
#                           outerConeAngle=np.pi/6.0)
#scene.add(light, pose=camera_pose)
#r = pyrender.OffscreenRenderer(400, 400)
#color, depth = r.render(scene)
#
#
#plt.figure()
#plt.subplot(1,2,1)
#plt.axis('off')
#plt.imshow(color)
#plt.subplot(1,2,2)
#plt.axis('off')
#plt.imshow(depth, cmap=plt.cm.gray_r)
#plt.show()