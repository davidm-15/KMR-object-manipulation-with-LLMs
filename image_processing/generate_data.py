import blenderproc as bproc
import os
import random
import glob
import numpy as np
# Init the scene

bproc.init()

base_dir = "object_models"
object_id = 1

for obj_folder in os.listdir(base_dir):
    mesh_folder = os.path.join(base_dir, obj_folder, "mesh")

    if not os.path.isdir(mesh_folder):
        continue

    # Find any .ply file
    ply_files = glob.glob(os.path.join(mesh_folder, "*.ply"))
    if len(ply_files) == 0:
        print(f"No .ply file found in {mesh_folder}, skipping.")
        continue
    ply_path = ply_files[0]

    # Optionally find a texture image (png or jpg)
    texture_files = glob.glob(os.path.join(mesh_folder, "*.png")) + glob.glob(os.path.join(mesh_folder, "*.jpg"))
    texture_path = texture_files[0] if texture_files else None

    # Load mesh
    objs = bproc.loader.load_obj(ply_path)

    for obj in objs:
        # Apply texture if present
        # if texture_path:
        #     mat = bproc.material.Material()
        #     mat.make_from_file(texture_path)
        #     obj.set_material(mat)

        # Set random pose and category ID
        obj.set_location(bproc.sampler.uniformSO3([-.5, -.5, 0.1], [.5, .5, 0.3]))
        obj.set_rotation_euler(bproc.sampler.uniformSO3([0, 0, 0], [3.14, 3.14, 3.14]))
        obj.set_cp("category_id", object_id)

        print(f"{obj=}")

    object_id += 1


# Set up camera
bproc.camera.set_resolution(512, 512)
rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
height = np.random.uniform(1.4, 1.8)
location = [1.2217, 0, 0]
cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
bproc.camera.add_camera_pose(cam2world_matrix)

# Light
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([2, 2, 3])
light.set_energy(1000)

# Render data
bproc.renderer.set_output_format("JPEG")
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id"])
data = bproc.renderer.render()
seg_data = bproc.renderer.render_segmap(map_by="category_id")


# Save coco annotations
bproc.writer.write_coco_annotations(
    os.path.join("output", "coco"),
    instance_segmaps=seg_data["segmap"],
    colors=data["colors"],
    color_map=seg_data["color_map"],
    append_to_existing_output=False
)

