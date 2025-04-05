# communication/megapose_inference_worker.py
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw # Added ImageDraw
import torch
import os
import traceback # For detailed error logging



# /mnt/proj3/open-29-7/.conda/envs/megapose/bin/python /mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/communication/megapose_inference_worker.py --image-path "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/foam brick/image_rgb.png" --object-name "foam brick" --bbox-json "[1141, 1187, 1522, 1742]" --output-file ./manual_worker_test_output.json --object-folder-root /mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects --camera-json /mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/image_processing/calibration_data/camera_intrinsics.json


# --- Add Megapose paths ---
# Assumes this script is in KMR_communication/communication
_CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CURRENT_FILE_DIR.parent # Go up two levels
MEGAPOSE_SRC_PATH = PROJECT_ROOT / "megapose6d" / "src" # Adjust if your megapose6d folder is elsewhere

if str(MEGAPOSE_SRC_PATH) not in sys.path:
    print(f"Worker: Adding Megapose src to path: {MEGAPOSE_SRC_PATH}")
    sys.path.append(str(MEGAPOSE_SRC_PATH))

try:
    # Import necessary components AFTER adjusting sys.path
    from megapose.scripts import run_inference_on_example
    from megapose.datasets.scene_dataset import CameraData, ObjectData
    from megapose.lib3d.transform import Transform
    from megapose.utils.logging import set_logging_level
    # Import the type for type hinting if available and needed
    try:
         from megapose.inference.types import PoseEstimatesType
    except ImportError:
         print("Worker: Warning - Could not import PoseEstimatesType, using Any for hinting.")
         from typing import Any
         PoseEstimatesType = Any # Define as Any if import fails

except ImportError as e:
    print(f"Worker: ERROR importing Megapose components: {e}")
    print(f"Worker: Please ensure Megapose is installed correctly and MEGAPOSE_SRC_PATH is correct: {MEGAPOSE_SRC_PATH}")
    print("Worker: Current sys.path:", sys.path)
    # Exit here won't help much as the calling server needs the error JSON
    # We'll rely on the main try-except block to report failure
    # sys.exit(1) # Avoid exiting here
    # Define dummies so the script doesn't crash immediately on import error
    run_inference_on_example = None
    CameraData = None
    ObjectData = None
    Transform = None
    PoseEstimatesType = Any


# --- Default Configuration ---
# Optional: Define default camera path if not passed via args
_DEFAULT_CAMERA_JSON_PATH = PROJECT_ROOT / "image_processing/calibration_data/camera_intrinsics.json"


def save_predictions_for_visualization(pose_estimates: PoseEstimatesType, file_path: Path):
    """
    Saves pose estimates to a JSON file in the format expected by
    visualization functions (list of ObjectData JSON).
    """
    # Implementation depends heavily on what run_inference_on_example.my_visualization expects.
    # Assuming it needs a list of ObjectData representations with TWO transforms.
    print(f"Worker (Vis Prep): Preparing data for visualization save at {file_path}")
    try:
        if not hasattr(pose_estimates, 'infos') or not hasattr(pose_estimates, 'poses'):
             print("Worker (Vis Prep): ERROR - pose_estimates object lacks 'infos' or 'poses' attribute.")
             return False # Indicate failure

        labels = pose_estimates.infos["label"].tolist() # Ensure it's a standard list
        poses_np = pose_estimates.poses.cpu().numpy()

        # Create ObjectData instances and convert to JSON serializable format
        object_data_list_serializable = [
            ObjectData(label=label, TWO=Transform(pose)).to_json()
            for label, pose in zip(labels, poses_np)
        ]

        # Note: The original megapose visualization often saves a dict like:
        # {"camera_data": cam_data.to_json(), "object_datas": obj_data_list_serializable}
        # Check if your my_visualization needs this structure or just the list.
        # Assuming it just needs the list of object data for now. Adapt if needed.
        output_json_string = json.dumps(object_data_list_serializable, indent=4)

        file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        file_path.write_text(output_json_string)
        print(f"Worker (Vis Prep): Saved temporary predictions for visualization to {file_path}")
        return True # Indicate success
    except Exception as e:
        print(f"Worker (Vis Prep): ERROR saving predictions for visualization: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return False # Indicate failure


def run_megapose(image_path: Path, object_name: str, bbox: list, camera_json_path: Path, object_folder_root: Path, output_file: Path, model_name: str, visualize: bool):
    """
    Loads data, runs inference, saves result, and optionally generates visualization.
    Writes results or error to output_file.
    """
    # Check if Megapose components loaded correctly
    if not all([run_inference_on_example, CameraData, ObjectData, Transform]):
         raise ImportError("Essential Megapose components failed to import. Cannot run inference.")

    temp_vis_outputs_dir = None
    temp_object_data_vis_path = None

    try:
        print(f"Worker: Loading image from {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image file not found: {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil, dtype=np.uint8)
        depth = None # Assuming RGB only for now, add depth loading if needed

        print(f"Worker: Loading camera data from {camera_json_path}")
        if not camera_json_path.is_file():
            raise FileNotFoundError(f"Camera intrinsics file not found: {camera_json_path}")
        camera_data = CameraData.from_json(camera_json_path.read_text())
        # Ensure resolution matches if camera_data has it
        if hasattr(camera_data, 'resolution') and tuple(camera_data.resolution) != image_pil.size[::-1]:
             print(f"Worker: WARNING - Image resolution {image_pil.size[::-1]} doesn't match camera data {camera_data.resolution}. Results may be inaccurate.")

        # --- Bbox format check ---
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Received invalid bbox format. Expected list [x1, y1, x2, y2].")
        # Ensure bbox values are numbers (int or float)
        bbox = [float(b) for b in bbox]
        print(f"Worker: Using bbox: {bbox}")


        object_data_list = [
            ObjectData(label=object_name, bbox_modal=np.array(bbox))
        ]
        print(f"Worker: Created ObjectData for '{object_name}' with bbox {bbox}")

        # --- Find object mesh folder ---
        # We need the specific object's folder (e.g., .../megapose_objects/foam_brick)
        object_mesh_folder = object_folder_root / object_name
        # if not object_mesh_folder.is_dir():
        #     # Attempt common variations like lowercase or replacing spaces
        #     variations_to_try = [
        #          object_name.lower(),
        #          object_name.replace(" ", "_"),
        #          object_name.lower().replace(" ", "_")
        #     ]
        #     found = False
        #     for variation in variations_to_try:
        #          potential_path = object_folder_root / variation
        #          if potential_path.is_dir():
        #               object_mesh_folder = potential_path
        #               print(f"Worker: Found object mesh folder using variation: {object_mesh_folder}")
        #               found = True
        #               break
        #     if not found:
        #          raise FileNotFoundError(f"Object mesh folder not found for '{object_name}' (or variations) inside root: {object_folder_root}")
        # else:
        #      print(f"Worker: Using object mesh folder: {object_mesh_folder}")

        print(f"Worker: Using object mesh folder: {object_mesh_folder}")

        print("Worker: Starting inference...")
        # Assuming my_inference exists and works like the original example's inference function
        pose_estimates: PoseEstimatesType = run_inference_on_example.my_inference(
            image=image,
            depth=depth,
            camera_data=camera_data,
            object_data=object_data_list,
            model_name=model_name,
            example_dir=object_mesh_folder # Pass the specific object's mesh folder
        )
        print("Worker: Inference finished.")

        # --- Process Result ---
        results_list = []
        pose_estimates_valid = False
        if pose_estimates is not None and hasattr(pose_estimates, 'infos') and hasattr(pose_estimates, 'poses'):
             if not pose_estimates.infos.empty:
                  pose_estimates_valid = True

        if pose_estimates_valid:
             print(f"Worker: Processing {len(pose_estimates.infos)} pose estimate(s)...")
             # Extract pose and label - make sure 'score' exists if needed, add dummy if not
             for i in range(len(pose_estimates.infos)):
                 info = pose_estimates.infos.iloc[i]
                 pose_matrix = pose_estimates.poses[i].cpu().numpy()
                 results_list.append({
                     "label": info["label"],
                     "score": float(info.get("score", 0.0)), # Safely get score, convert to float
                     "pose": pose_matrix.tolist() # Convert numpy array to list for JSON
                 })
             result_data = {"poses": results_list}
             print(f"Worker: Successfully processed {len(results_list)} pose(s).")
        else:
             result_data = {"poses": [], "message": "No valid pose estimates returned by Megapose."} # Use 'message' instead of 'error' for no poses
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
                #    Megapose example often puts this in example_dir/outputs
                temp_vis_outputs_dir = object_mesh_folder / "outputs_vis_temp" # Use unique temp name
                temp_vis_outputs_dir.mkdir(exist_ok=True)
                temp_object_data_vis_path = temp_vis_outputs_dir / "object_data.json"

                # 3. Save predictions temporarily for the visualization function to read
                save_success = save_predictions_for_visualization(pose_estimates, temp_object_data_vis_path)

                if save_success:
                    # 4. Call the visualization function from run_inference_on_example
                    #    Ensure 'my_visualization' exists and takes these arguments.
                    print(f"Worker (Vis): Calling my_visualization with example_dir={object_mesh_folder}, out_path={vis_output_dir}")
                    # Check if my_visualization exists before calling
                    if hasattr(run_inference_on_example, 'my_visualization'):
                        vis_results_dict = run_inference_on_example.my_visualization(
                            example_dir=object_mesh_folder, # Dir containing meshes and temp 'outputs/object_data.json'
                            out_path=vis_output_dir,        # Where to save PNGs
                            # Pass any other required args, e.g., maybe it needs the camera data again?
                            SavePredictions=True # Seems like an arg from the original script
                        )
                        # Add path info to the main result if needed
                        result_data["visualization_output_dir"] = str(vis_output_dir) # Inform server/client
                        print(f"Worker (Vis): Visualization images should be saved to {vis_output_dir}")
                    else:
                        print("Worker (Vis): ERROR - 'my_visualization' function not found in run_inference_on_example module.")
                        result_data["visualization_error"] = "Visualization function not found."

                else:
                     print("Worker (Vis): Skipping visualization call due to error saving predictions.")
                     result_data["visualization_error"] = "Failed to prepare data for visualization."

            except Exception as vis_e:
                # Log clearly that visualization failed but don't stop the whole process
                print(f"Worker: WARNING - Visualization generation failed: {type(vis_e).__name__} - {vis_e}\n{traceback.format_exc()}", file=sys.stderr)
                result_data["visualization_error"] = f"Visualization failed: {str(vis_e)}"

            finally:
                # 5. Clean up the temporary visualization input file/dir regardless of success
                if temp_object_data_vis_path and temp_object_data_vis_path.exists():
                    try:
                        temp_object_data_vis_path.unlink()
                        print(f"Worker (Vis): Cleaned up temporary file {temp_object_data_vis_path}")
                    except OSError as clean_e:
                        print(f"Worker (Vis): WARNING - Could not remove temp vis file {temp_object_data_vis_path}: {clean_e}", file=sys.stderr)
                if temp_vis_outputs_dir and temp_vis_outputs_dir.exists():
                    try:
                        # Remove the directory and its contents
                        import shutil
                        shutil.rmtree(temp_vis_outputs_dir)
                        print(f"Worker (Vis): Cleaned up temporary dir {temp_vis_outputs_dir}")
                    except OSError as clean_e:
                        print(f"Worker (Vis): WARNING - Could not remove temp vis dir {temp_vis_outputs_dir}: {clean_e}", file=sys.stderr)


        # --- Save Final JSON Result (always happens) ---
        print(f"Worker: Writing final JSON result to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        print("Worker: Successfully completed final JSON output.")

    except Exception as e:
        # --- Error Handling: Write error to output file ---
        error_message = f"Worker Error: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr) # Log error to stderr
        try:
            # Attempt to write error to the designated output file
            output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            with open(output_file, 'w') as f:
                json.dump({"error": f"Worker Error: {type(e).__name__} - {e}"}, f) # Simpler error in JSON
            print(f"Worker: Wrote error message to {output_file}")
        except Exception as write_e:
            # If writing the error file fails, just log it
            print(f"Worker: CRITICAL - Failed to write error to output file {output_file}: {write_e}", file=sys.stderr)
        sys.exit(1) # Exit with non-zero code to signal failure to the server


if __name__ == "__main__":
    # Set logging level (consider making this an argument too)
    set_logging_level("warn") # Options: 'debug', 'info', 'warn', 'error', 'critical'

    parser = argparse.ArgumentParser(description="MegaPose Inference Worker")
    parser.add_argument("--image-path", required=True, type=Path, help="Path to the input image file")
    parser.add_argument("--object-name", required=True, type=str, help="Label of the object (must match folder name in object root)")
    parser.add_argument("--bbox-json", required=True, type=str, help="Bounding box as a JSON string list (e.g., '[xmin, ymin, xmax, ymax]')")
    parser.add_argument("--camera-json", type=Path, default=_DEFAULT_CAMERA_JSON_PATH, help="Path to camera intrinsics JSON")
    parser.add_argument("--object-folder-root", required=True, type=Path, help="Path to the root directory containing object mesh folders (e.g., 'megapose_objects')")
    parser.add_argument("--output-file", required=True, type=Path, help="Path to save the JSON output")
    parser.add_argument("--model-name", type=str, default="megapose-1.0-RGB-multi-hypothesis", help="Name of the MegaPose model to use")
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualization images")

    args = parser.parse_args()

    # --- Bbox parsing ---
    try:
        bbox_list = json.loads(args.bbox_json)
        # Basic validation performed inside run_megapose now
    except json.JSONDecodeError as e:
        errmsg = f"Worker: Error parsing bbox JSON: {e}. Input was: {args.bbox_json}"
        print(errmsg, file=sys.stderr)
        # Write error to output file before exiting
        try:
             args.output_file.parent.mkdir(parents=True, exist_ok=True)
             args.output_file.write_text(json.dumps({"error": errmsg}))
        except Exception: pass # Ignore errors writing error file
        sys.exit(1)

    print("Worker: Parsed arguments successfully.")
    # --- Call the main function ---
    run_megapose(
        image_path=args.image_path,
        object_name=args.object_name,
        bbox=bbox_list,
        camera_json_path=args.camera_json,
        object_folder_root=args.object_folder_root,
        output_file=args.output_file,
        model_name=args.model_name,
        visualize=args.visualize
    )
    print("Worker: Script finished.")
    sys.exit(0) # Explicitly exit with 0 on success