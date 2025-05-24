# place me in megapose/scripts/run_inference_on_example.py

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import torch

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


logger = get_logger(__name__)


def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((Path("image_processing/calibration_data/camera_intrinsics.json")).read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


# def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
#     print("EXAMPLE DIR     -----------AAAAAAAAAA-----------   ", example_dir)
#     rigid_objects = []
#     mesh_units = "mm"
#     object_dirs = (example_dir / "mesh").iterdir()
#     print("object_dirs", object_dirs)
#     for object_dir in object_dirs:
#         print("object_dir", object_dir)
#         label = object_dir.name
#         print("label", label)
#         mesh_path = None
#         for fn in object_dir.glob("*"):
#             print("fn", fn)
#             if fn.suffix in {".obj", ".ply"}:
#                 assert not mesh_path, f"there multiple meshes in the {label} directory"
#                 mesh_path = fn
#         assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
#         rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
#         # TODO: fix mesh units
#     rigid_object_dataset = RigidObjectDataset(rigid_objects)
#     return rigid_object_dataset

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    print("EXAMPLE DIR:", example_dir)
    rigid_objects = []
    mesh_units = "mm"
    
    # Check for PLY files directly in the mesh directory
    mesh_dir = example_dir / "mesh"
    mesh_files = list(mesh_dir.glob("*.ply"))
    
    if not mesh_files:
        # Also check for obj files if no ply files found
        mesh_files = list(mesh_dir.glob("*.obj"))
    
    if mesh_files:
        # Use the first mesh file found
        mesh_path = mesh_files[0]
        # Extract label from the filename (without extension)
        label = mesh_path.stem
        print(f"Found mesh: {mesh_path}, label: {label}")
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    else:
        # Fall back to original directory-per-object approach
        object_dirs = [d for d in mesh_dir.iterdir() if d.is_dir()]
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldn't find a obj or ply mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_object_dataset_all(example_dir: Path) -> RigidObjectDataset:
    print("EXAMPLE DIR:", example_dir)
    rigid_objects = []
    mesh_units = "mm"
    
    # Find all subdirectories in the example_dir
    subdirectories = [d for d in example_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirectories:
        # Look for mesh folder in each subdirectory
        mesh_dir = subdir / "mesh"
        if not mesh_dir.exists() or not mesh_dir.is_dir():
            print(f"Skipping {subdir}: no mesh directory found")
            continue
            
        # Find all PLY files in the mesh directory
        mesh_files = list(mesh_dir.glob("*.ply"))
        
        if not mesh_files:
            print(f"No PLY files found in {mesh_dir}")
            continue
            
        for mesh_path in mesh_files:
            # Extract label from the filename (without extension)
            label = mesh_path.stem
            print(f"label: {label}")
            print(f"Found mesh: {mesh_path}, label: {label}")
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    
    if not rigid_objects:
        print("Warning: No rigid objects were loaded!")
        
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset




def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def my_inference(
        image: np.ndarray,
        depth: np.ndarray,
        camera_data: CameraData,
        object_data: List[ObjectData],
        model_name: str,
        example_dir: Path
) -> PoseEstimatesType:
    print("G"*100)
    
    observation = ObservationTensor.from_numpy(image, depth, camera_data.K).cuda()

    detections = make_detections_from_object_data(object_data).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    print("H"*100)
    model_info = NAMED_MODELS[model_name]
    logger.info(f"Running inference.")
    print("I"*100)
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    print("J"*100)
    print("-"*100)
    print(f"{output=}")
    return output

def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:
    
    print(">"*100)
    print(f"{example_dir=}")

    model_info = NAMED_MODELS[model_name]
    print(f"{model_info=}")

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    print(f"{observation=}")

    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    print("-"*100)
    save_predictions(example_dir, output)
    return


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def my_visualization(
    example_dir: Path,
    out_path: Path,
    **kwargs,
) -> dict:
    
    SavePredictions = kwargs.get("SavePredictions", True)
    time = kwargs.get("time", "00_00_00")
    
    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    # Convert everything to appropriate format for matplotlib
    rgb_plot = rgb.copy()
    if rgb_plot.max() > 1.0:  # Normalize if needed
        rgb_plot = rgb_plot / 255.0
    
    mesh_overlay = renderings.rgb.copy()
    if mesh_overlay.max() > 1.0:
        mesh_overlay = mesh_overlay / 255.0

    # Blend the mesh overlay with the original image
    alpha = 0.5  # Adjust transparency level
    mesh_overlay = (alpha * mesh_overlay + (1 - alpha) * rgb / 255.0).clip(0, 1)
    
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    if contour_overlay.max() > 1.0:
        contour_overlay = contour_overlay / 255.0

    # Save images if needed
    vis_dir = out_path
    vis_dir.mkdir(exist_ok=True)
    
    if SavePredictions:
        plt.imsave(vis_dir / f"mesh_overlay_{time}.png", mesh_overlay)
        plt.imsave(vis_dir / f"contour_overlay_{time}.png", contour_overlay)

    # Return images as a dictionary of NumPy arrays
    return {
        "rgb": rgb_plot,
        "mesh_overlay": mesh_overlay,
        "contour_overlay": contour_overlay,
    }


def make_output_visualization(
    example_dir: Path,
) -> None:

    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    logger.info(f"Wrote visualizations to {vis_dir}.")
    return


# def make_mesh_visualization(RigidObject) -> List[Image]:
#     return


# def make_scene_visualization(CameraData, List[ObjectData]) -> List[Image]:
#     return


# def run_inference(example_dir, use_depth: bool = False):
#     return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    parser.add_argument("--my-inference", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        print("Running inference")
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)

    if args.my_inference:
        print("Running my inference")
        print(f"{example_dir=}")
        example_dir = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/megapose6d/local_data/examples/mustard-bottle")
        image, depth, camera_data = load_observation(example_dir, load_depth=False)
        object_data = load_object_data(example_dir / "inputs" / "object_data.json")
        my_inference(image, depth, camera_data, object_data, args.model, example_dir)

