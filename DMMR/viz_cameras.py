from core.utils.visualization3d import Visualization
from core.utils.module_utils import load_camera_para, add_camera_mesh
from render.visualization import VisOpen3D
import numpy as np
import open3d.visualization.gui as gui
import open3d as o3d
import sys
import os
import time
import keyboard
import time
import yaml


Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh
load_mesh = o3d.io.read_triangle_mesh



def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh



def add_mesh(viz, path, color): 
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    viz.add_geometry(mesh)
    mesh.paint_uniform_color(color)
    return mesh

def  add_floor_mesh(viz, path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    viz.add_geometry(mesh)
    return mesh


def update_mesh(viz, path, color, old_mesh):
    mesh = o3d.io.read_triangle_mesh(path)

    old_mesh.vertices = Vector3dVector(mesh.vertices)
    old_mesh.triangles = mesh.triangles
    #qqold_mesh.subdivide_loop(number_of_iterations=1)
    viz.update_geometry(old_mesh)
    
    return old_mesh


if __name__ == '__main__': 
    #width=1024 
    #height=768 
    width=960 
    height=620 
    file_path = os.path.join(os.getcwd(), 'DMMR')
    
    mother_paths = []
    child_paths = []
    mesh_colors = [[0.6, 1, 0.6], [1, 0.6, 0.6]] 

    cam_color = [[1, 0, 0], [.95, 1, 0], [0, 0, .95], [0, 1, 0]]           
    is_camera_on_in_batch = []
    
    mesh_data_path = os.path.join(file_path, 'output', 'meshes')
    data_path = []

    det_path, batch_files, _ = next(os.walk(mesh_data_path))
    for batch in batch_files:
        data_path.append(os.path.join(det_path, batch))
        print(batch)
        active_cams = str(batch).split('_')[-1]
        is_camera_on = [True, True, True, True]

        if active_cams[0] == 'A':
            is_camera_on_in_batch.append(is_camera_on)
        elif active_cams[0] == 'N':
            is_camera_on[int(active_cams[1])] = False
            is_camera_on_in_batch.append(is_camera_on)
    
    
    print(is_camera_on_in_batch)



    with open(os.path.join(file_path, 'cfg_files', 'fit_smpl.yaml'), 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)
    file.close()

    if not yaml_data['opt_cam']:
        cam_path = os.path.join(file_path, 'data', 'YOUth_camparams', 'camparams.txt')
    else:#################
        cam_path = os.path.join(file_path, 'output', 'camparams', str(yaml_data['frames']).zfill(5)+'.txt')
        

    extris, intris = load_camera_para(cam_path)

    # create window
    vis = VisOpen3D(width=width, height=height, visible=True)
    viewr = vis.get_vis()
    

    plane = o3d.geometry.TriangleMesh.create_box(10, 10, depth=1e-6)
    plane.paint_uniform_color([.2,.3,.4])
    plane.translate([0, 0, 0])
    #vis.add_geometry(plane)

    opt = viewr.get_render_option()
    opt.background_color = np.asarray([0.899, 0.899, 0.899])

    translation = []
    camera_mesh_list = []
    cam_line_list = []
    for cam_idx, cam in enumerate(extris):
        translation.append(cam.T)
        cam = add_camera_mesh(cam, camerascale=0.1)
        line_list = vis.visualize_cameras(cam.T, cam_color[cam_idx])
        cam_line_list.append(line_list)

    batch_frame_lenght = []
    batch_frame = 0

    for mesh_path in data_path:
        for mesh in os.listdir(mesh_path):
            if mesh.endswith('00.obj'):
                batch_frame += 1
                mother_paths.append(os.path.join(mesh_path, mesh))
            elif mesh.endswith('01.obj'): 
                child_paths.append(os.path.join(mesh_path, mesh))
        if mesh:
            batch_frame_lenght.append(batch_frame)
    
    if batch_frame_lenght[0] == len(mother_paths):
        batch_frame_lenght = []
    
    #initial mother mesh
    m_00 = add_mesh(vis, mother_paths[0], mesh_colors[0])
    #initial child mesh 
    m_01 = add_mesh(vis, child_paths[0], mesh_colors[1])
    m_id = 1 
    print(f'Frame #{m_id}')

    #Set initial view to camera 0
    vis.update_view_point(intris[0], extris[0])
    vis.poll_events()
    vis.update_renderer()

    in_batch = 0
    last_batch = -1 if batch_frame_lenght else 0

    while not keyboard.is_pressed("q"):
    #while False:
        
        vis.poll_events()
        if batch_frame_lenght:
            if m_id < batch_frame_lenght[0]:
                in_batch = 0
            if m_id > batch_frame_lenght[0] and m_id < batch_frame_lenght[1]:
                in_batch = 1
            if m_id > batch_frame_lenght[1] and m_id <= batch_frame_lenght[2]:
                in_batch = 2

        if last_batch != in_batch:
            for c_id in range(len(cam_line_list)):
                if is_camera_on_in_batch[in_batch][c_id]:
                    vis.update_camera_color(cam_line_list[c_id], cam_color[c_id])
                else: 
                    vis.update_camera_color(cam_line_list[c_id], opt.background_color)
            last_batch = in_batch


        if keyboard.is_pressed("a"):
            if m_id >= len(mother_paths):
                m_id = 0 
            m_00 = update_mesh(vis, mother_paths[m_id], mesh_colors[0], m_00)
            m_01 = update_mesh(vis, child_paths[m_id], mesh_colors[1], m_01)
            m_id += 1
            print(f'Frame #{m_id}')
            time.sleep(.15)

        if keyboard.is_pressed("b"):
            if m_id < 0:
                m_id = len(mother_paths) - 1
            m_00 = update_mesh(vis, mother_paths[m_id], mesh_colors[0], m_00)
            m_01 = update_mesh(vis, child_paths[m_id], mesh_colors[1], m_01)
            m_id -= 1
            print(f'Frame #{m_id}')
            time.sleep(.15)

        if keyboard.is_pressed('space'):
            if m_id >= len(mother_paths):
                m_id = 0 
            m_00 = update_mesh(vis, mother_paths[m_id], mesh_colors[0], m_00)
            m_01 = update_mesh(vis, child_paths[m_id], mesh_colors[1], m_01)
            m_id += 1
            print(f'Frame #{m_id}')

        if keyboard.is_pressed('r'):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)

        if keyboard.is_pressed('s'):
            intrinsic = vis.get_view_point_intrinsics()
            extrinsic = vis.get_view_point_extrinsics()
            vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
            time.sleep(.2)


        if keyboard.is_pressed("1"):
            if batch_frame_lenght:
                if is_camera_on_in_batch[in_batch][0]:
                    vis.update_view_point(intris[0], extris[0])
                    time.sleep(.15)
            else:
                vis.update_view_point(intris[0], extris[0])
                time.sleep(.15)

        if keyboard.is_pressed("2"):
            if batch_frame_lenght:
                if is_camera_on_in_batch[in_batch][1]:
                    print('In view 1')
                    vis.update_view_point(intris[1], extris[1])
                    time.sleep(.15)
            else:
                vis.update_view_point(intris[1], extris[1])
                time.sleep(.15)

        if keyboard.is_pressed("3"):
            if batch_frame_lenght:
                if is_camera_on_in_batch[in_batch][2]:
                    print('In view 2')
                    vis.update_view_point(intris[2], extris[2])
                    time.sleep(.15)
            else:
                vis.update_view_point(intris[2], extris[2])
                time.sleep(.15)

        if keyboard.is_pressed("4"):
            if batch_frame_lenght:
                if is_camera_on_in_batch[in_batch][3]:
                    print('In view 3')
                    vis.update_view_point(intris[3], extris[3])
                    time.sleep(.15) 
            else:
                vis.update_view_point(intris[3], extris[3])
                time.sleep(.15) 

    #vis.close()
    vis.poll_events()
    vis.destroy_window()
    vis.__del__()
    sys.exit()
