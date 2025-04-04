# megapose_inference_worker.py
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import os
import traceback  # For detailed error logging

# --- Add Megapose paths if necessary ---
# Determine the base directory of your project
# Adjust this path based on where megapose_inference_worker.py is located
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Adjust if needed
MEGAPOSE_SRC_PATH = PROJECT_ROOT / "megapose6d" / "src"
if str(MEGAPOSE_SRC_PATH) not in sys.path:
     print(f"Adding Megapose src to path: {MEGAPOSE_SRC_PATH}")
     sys.path.append(str(MEGAPOSE_SRC_PATH))
# It seems run_inference_on_example is already in the path for you,
# but explicitly adding can sometimes help prevent issues.
# You might need to adjust how you import it based on your structure.
try:
    from megapose.scripts import run_inference_on_example
    from megapose.datasets.scene_dataset import CameraData, ObjectData
    from megapose.lib3d.transform import Transform
    from megapose.utils.logging import set_logging_level
    # --- ADD THIS LINE ---
    from megapose.inference.types import PoseEstimatesType
    # --- END ADD ---
except ImportError as e:
    print(f"Error importing Megapose components: {e}")
    # Include PoseEstimatesType in the error message if it fails
    if 'PoseEstimatesType' in str(e):
         print("Could not import 'PoseEstimatesType'. Check megapose.inference.types path.")
    print("Current sys.path:", sys.path)
    sys.exit(1)


try:
    # Assuming PoseEstimatesType is defined or used within run_inference_on_example context
    # If not, you might need to import it directly if it's defined elsewhere
    # e.g., from megapose.inference.types import PoseEstimatesType
    pass # Placeholder assuming it's okay
except ImportError:
    print("Warning: Could not explicitly import PoseEstimatesType")
    # Define a dummy type if needed for type hinting, or remove hint
    from typing import Any
    PoseEstimatesType = Any

# --- Default Configuration ---
# It's better to pass these via args, but defaults can be useful
DEFAULT_MODEL_NAME = "megapose-1.0-RGB-multi-hypothesis"
# You MUST provide the correct path to your camera intrinsics
DEFAULT_CAMERA_JSON_PATH = PROJECT_ROOT / "image_processing/calibration_data/camera_intrinsics.json"

def run_megapose(image_path: Path, object_name: str, bbox: list, camera_json_path: Path, object_folder_root: Path, output_file: Path, model_name: str, visualize: bool): # Added visualize flag
    """
    Loads data, runs inference, saves result, and optionally generates visualization.
    """
    temp_vis_outputs_dir = None # For cleanup tracking
    temp_object_data_path = None # For cleanup tracking

    try:
        print(f"Worker: Loading image from {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        depth = None

        print(f"Worker: Loading camera data from {camera_json_path}")
        if not camera_json_path.is_file():
             raise FileNotFoundError(f"Camera intrinsics file not found: {camera_json_path}")
        camera_data = CameraData.from_json(camera_json_path.read_text())

        object_data_list = [
            ObjectData(label=object_name, bbox_modal=np.array(bbox))
        ]
        print(f"Worker: Created ObjectData for '{object_name}' with bbox {bbox}")

        object_folder = object_folder_root / object_name
        if not object_folder.is_dir():
             raise FileNotFoundError(f"Object mesh folder not found: {object_folder}")
        print(f"Worker: Using object mesh folder: {object_folder}")


        print("Worker: Starting inference...")
        pose_estimates = run_inference_on_example.my_inference(
            image=image,
            depth=depth,
            camera_data=camera_data,
            object_data=object_data_list,
            model_name=model_name,
            example_dir=object_folder
        )
        print("Worker: Inference finished.")

        # --- Process Result (same as before) ---
        results_list = []
        pose_estimates_valid = pose_estimates is not None and hasattr(pose_estimates, 'infos') and not pose_estimates.infos.empty
        if pose_estimates_valid:
             # Extract pose and label - make sure 'score' exists if needed, add dummy if not
             for i in range(len(pose_estimates.infos)):
                 info = pose_estimates.infos.iloc[i]
                 pose_matrix = pose_estimates.poses[i].cpu().numpy()
                 results_list.append({
                     "label": info["label"],
                     # Include score if available in pose_estimates.infos, else omit or default
                     "score": info.get("score", 0.0), # Safely get score, default 0.0
                     "pose": pose_matrix.tolist()
                 })
             result_data = {"poses": results_list}
             print(f"Worker: Found {len(results_list)} pose(s).")
        else:
             result_data = {"poses": [], "error": "No valid pose estimates returned"}
             print("Worker: No valid pose estimates returned by Megapose.")


        # --- Visualization (Optional) ---
        if visualize and pose_estimates_valid: # Only run if visualize flag is true AND we have poses
            print("Worker: Generating visualization...")
            try:
                # 1. Define output dir for visualization images (next to json output)
                vis_output_dir = Path(output_file.parent / f"{output_file.stem}_visualizations")
                vis_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Worker (Vis): Saving visualizations to {vis_output_dir}")

                # 2. Define temporary location for the object_data.json needed by my_visualization
                #    Place it inside the object_folder structure as expected.
                temp_vis_outputs_dir = object_folder / "outputs"
                temp_vis_outputs_dir.mkdir(exist_ok=True)
                temp_object_data_path = temp_vis_outputs_dir / "object_data.json"

                # 3. Save predictions temporarily for the visualization function to read
                save_predictions_for_visualization(pose_estimates, temp_object_data_path)

                # 4. Call the visualization function from run_inference_on_example
                #    Ensure 'my_visualization' exists and takes these arguments.
                print(f"Worker (Vis): Calling my_visualization with example_dir={object_folder}, out_path={vis_output_dir}")
                vis_results_dict = run_inference_on_example.my_visualization(
                    example_dir=object_folder, # Dir containing meshes and temp 'outputs/object_data.json'
                    out_path=vis_output_dir,   # Where to save PNGs
                    SavePredictions=True       # Tell it explicitly to save images
                )
                # You could potentially do something with vis_results_dict if needed,
                # but for now we just care about the saved files.
                print(f"Worker (Vis): Visualization images saved successfully to {vis_output_dir}")

            except AttributeError:
                 print("Worker (Vis): ERROR - 'my_visualization' function not found in run_inference_on_example module.", file=sys.stderr)
            except Exception as vis_e:
                # Log clearly that visualization failed but don't stop the whole process
                print(f"Worker: WARNING - Visualization generation failed: {type(vis_e).__name__} - {vis_e}\n{traceback.format_exc()}", file=sys.stderr)

            finally:
                # 5. Clean up the temporary visualization input file/dir regardless of success
                pass
                # if temp_object_data_path and temp_object_data_path.exists():
                #     try:
                #         temp_object_data_path.unlink()
                #         print(f"Worker (Vis): Cleaned up temporary file {temp_object_data_path}")
                #     except OSError as clean_e:
                #         print(f"Worker (Vis): WARNING - Could not remove temp vis file {temp_object_data_path}: {clean_e}", file=sys.stderr)
                # if temp_vis_outputs_dir and temp_vis_outputs_dir.exists():
                #     try:
                #          # Only remove if empty, check first to be safe
                #          if not any(temp_vis_outputs_dir.iterdir()):
                #               temp_vis_outputs_dir.rmdir()
                #               print(f"Worker (Vis): Cleaned up temporary dir {temp_vis_outputs_dir}")
                #          else:
                #               print(f"Worker (Vis): WARNING - Temp vis dir {temp_vis_outputs_dir} not empty, not removing.")
                #     except OSError as clean_e:
                #         print(f"Worker (Vis): WARNING - Could not remove temp vis dir {temp_vis_outputs_dir}: {clean_e}", file=sys.stderr)


        # --- Save Final JSON Result (always happens) ---
        print(f"Worker: Writing final JSON result to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        print("Worker: Successfully completed final JSON output.")

    except Exception as e:
        # --- Error Handling (same as before) ---
        error_message = f"Worker Error: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)
        try:
            with open(output_file, 'w') as f:
                json.dump({"error": error_message}, f)
        except Exception as write_e:
            print(f"Worker: Failed to write error to output file: {write_e}", file=sys.stderr)
        sys.exit(1) # Exit with error code

    # --- No finally block needed here specifically for cleanup, it's handled in the vis block ---


# --- Helper function to save predictions in the format my_visualization expects ---
def save_predictions_for_visualization(pose_estimates: PoseEstimatesType, file_path: Path):
    """
    Saves pose estimates to a JSON file in the format expected by
    visualization functions (list of ObjectData JSON).
    """
    try:
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        # Create ObjectData instances (without score, as before)
        # Note: my_visualization might implicitly need TWC=Transform(np.eye(4)) on CameraData later
        # and TWO from the predictions.
        object_data_list = [
            ObjectData(label=label, TWO=Transform(pose))
            for label, pose in zip(labels, poses)
        ]
        object_data_json_serializable = [x.to_json() for x in object_data_list]
        output_json_string = json.dumps(object_data_json_serializable, indent=4)

        file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        file_path.write_text(output_json_string)
        print(f"Worker (Vis): Saved temporary predictions for visualization to {file_path}")
    except Exception as e:
        print(f"Worker (Vis): ERROR saving predictions for visualization: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise # Re-raise the exception to indicate failure

if __name__ == "__main__":
    set_logging_level("warn") # Adjust logging level (e.g., 'info' or 'debug' for more worker details)

    parser = argparse.ArgumentParser(description="MegaPose Inference Worker")
    # ... (existing arguments) ...
    parser.add_argument("--image-path", required=True, type=Path, help="Path to the input image file")
    parser.add_argument("--object-name", required=True, type=str, help="Label of the object")
    parser.add_argument("--bbox-json", required=True, type=str, help="Bounding box as a JSON string list (e.g., '[xmin, ymin, xmax, ymax]')")
    parser.add_argument("--camera-json", type=Path, default=DEFAULT_CAMERA_JSON_PATH, help="Path to camera intrinsics JSON")
    parser.add_argument("--object-folder-root", required=True, type=Path, help="Path to the root directory containing object mesh folders (e.g., 'megapose_objects')")
    parser.add_argument("--output-file", required=True, type=Path, help="Path to save the JSON output")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the MegaPose model to use")
    # --- Add visualization flag ---
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualization images")

    args = parser.parse_args()

    # --- Bbox parsing (same as before) ---
    try:
        bbox_list = json.loads(args.bbox_json)
        if not isinstance(bbox_list, list) or len(bbox_list) != 4:
            raise ValueError("Bbox JSON must be a list of 4 numbers.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing bbox JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print("Worker: Parsed arguments successfully.")
    # --- Call run_megapose with the visualize flag ---
    run_megapose(
        image_path=args.image_path,
        object_name=args.object_name,
        bbox=bbox_list,
        camera_json_path=args.camera_json,
        object_folder_root=args.object_folder_root,
        output_file=args.output_file,
        model_name=args.model_name,
        visualize=args.visualize # Pass the flag here
    )
