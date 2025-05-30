import tkinter as tk
import tkinter.ttk as ttk
import numpy as np

# --- Use Relative Imports ---
from utils import config
from . import kuka_api as api
import json
import os
from . import sequences # If GUI calls sequence functions

# run me with python -m KMR_communication.main --mode gui


def adjust_joint(joint_label: str, delta_deg: float):
    """Gets current joints, calculates new position, and moves the joint."""
    print(f"Adjusting joint {joint_label} by {delta_deg} degrees")
    current_position = api.get_iiwa_joint_position()
    if current_position:
        # Convert delta from degrees to radians for calculation
        delta_rad = np.deg2rad(delta_deg)
        new_position = current_position.copy() # Create a copy
        new_position[joint_label] += delta_rad

        # Call goto_joint with all joint values from the new_position dict
        # Ensure the order matches the function definition A1-A7
        try:
            api.goto_joint(
                new_position["A1"], new_position["A2"], new_position["A3"],
                new_position["A4"], new_position["A5"], new_position["A6"],
                new_position["A7"],
                speed=0.3 # Use a slightly faster speed for jogging
            )
        except KeyError as e:
             print(f"Error: Joint key missing in current position data: {e}")
        except Exception as e:
             print(f"Error during joint adjustment: {e}")
    else:
        print("Could not get current joint positions to adjust.")


def create_gui(camera_handler): # Pass camera_handler if needed by GUI actions
    """Creates and runs the Tkinter GUI."""
    root = tk.Tk()
    root.title("KUKA KMR IIWA Controller")
    root.geometry("650x750") # Slightly wider for joint controls

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # --- Movement Page ---
    movement_frame = ttk.Frame(notebook, padding="10")
    notebook.add(movement_frame, text="KMR Movement")

    control_frame = ttk.Frame(movement_frame)
    control_frame.pack(pady=10)

    # KMR Base Controls
    ttk.Button(control_frame, text="↑ (X+)", width=6, command=lambda: api.move(0.1, 0, 0)).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(control_frame, text="← (Y+)", width=6, command=lambda: api.move(0, 0.1, 0)).grid(row=1, column=0, padx=5, pady=5)
    ttk.Button(control_frame, text="→ (Y-)", width=6, command=lambda: api.move(0, -0.1, 0)).grid(row=1, column=2, padx=5, pady=5)
    ttk.Button(control_frame, text="↓ (X-)", width=6, command=lambda: api.move(-0.1, 0, 0)).grid(row=2, column=1, padx=5, pady=5)
    ttk.Button(control_frame, text="↺ (Th+)", width=6, command=lambda: api.move(0, 0, 0.1)).grid(row=1, column=3, padx=5, pady=5)
    ttk.Button(control_frame, text="↻ (Th-)", width=6, command=lambda: api.move(0, 0, -0.1)).grid(row=1, column=4, padx=5, pady=5)

    ttk.Separator(movement_frame, orient='horizontal').pack(fill='x', pady=15)

    # KMR Predefined Locations
    loc_frame = ttk.Frame(movement_frame)
    loc_frame.pack(pady=10)
    ttk.Label(loc_frame, text="Move to Location:").pack(side=tk.LEFT, padx=5)
    loc_entry = ttk.Entry(loc_frame, width=5)
    loc_entry.pack(side=tk.LEFT, padx=5)
    loc_entry.insert(0, "1") # Default value
    ttk.Button(loc_frame, text="Go", command=lambda: api.move_to_location(int(loc_entry.get()))).pack(side=tk.LEFT, padx=5)


    # --- IIWA Arm Control Page ---
    iiwa_frame = ttk.Frame(notebook, padding="10")
    notebook.add(iiwa_frame, text="IIWA Control")

    # === Go To Cartesian Position Section ===
    pos_frame = ttk.LabelFrame(iiwa_frame, text="Go To Cartesian Position", padding="10")
    pos_frame.pack(fill="x", pady=10)

    labels_pos = ["X (mm)", "Y (mm)", "Z (mm)", "A", "B", "C"]
    entries_pos = {}
    for i, label in enumerate(labels_pos):
        ttk.Label(pos_frame, text=label+":", width=7).grid(row=i//3, column=(i%3)*2, sticky="w", padx=2, pady=2)
        entry = ttk.Entry(pos_frame, width=8)
        entry.grid(row=i//3, column=(i%3)*2 + 1, padx=2, pady=2)
        entries_pos[label.split(" ")[0]] = entry # Use key like 'X', 'A'

    degree_mode_position = tk.BooleanVar(value=True)
    ttk.Checkbutton(pos_frame, text="Angles in Degrees", variable=degree_mode_position).grid(row=0, column=6, padx=10)

    def _goto_cartesian():
        try:
            x = float(entries_pos['X'].get())
            y = float(entries_pos['Y'].get())
            z = float(entries_pos['Z'].get())
            a = float(entries_pos['A'].get())
            b = float(entries_pos['B'].get())
            c = float(entries_pos['C'].get())
            if degree_mode_position.get():
                a, b, c = np.deg2rad(a), np.deg2rad(b), np.deg2rad(c)
            api.goto_position(x, y, z, a, b, c) # Assumes A,B,C are ZYX intrinsic or similar KUKA convention
        except ValueError:
            print("Invalid input for Cartesian position.")
        except Exception as e:
             print(f"Error in Go To Position: {e}")

    ttk.Button(pos_frame, text="Go To Position", command=_goto_cartesian).grid(row=1, column=6, pady=10, padx=10)
    ttk.Button(iiwa_frame, text="Go Home", command=api.go_home).pack(pady=5)


    # === Go To Joint Position Section ===
    joint_goto_frame = ttk.LabelFrame(iiwa_frame, text="Go To Joint Position", padding="10")
    joint_goto_frame.pack(fill="x", pady=10)

    joint_labels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    joint_goto_entries = {}
    for i, label in enumerate(joint_labels):
        ttk.Label(joint_goto_frame, text=label+":", width=4).grid(row=0, column=i*2, padx=2, pady=2)
        entry = ttk.Entry(joint_goto_frame, width=7)
        entry.grid(row=0, column=i*2+1, padx=2, pady=2)
        joint_goto_entries[label] = entry

    degree_mode_joint = tk.BooleanVar(value=True)
    ttk.Checkbutton(joint_goto_frame, text="Angles in Degrees", variable=degree_mode_joint).grid(row=1, column=0, columnspan=4, pady=5)

    def _goto_joint_pos():
        try:
            joint_values = []
            for label in joint_labels:
                val = float(joint_goto_entries[label].get())
                if degree_mode_joint.get():
                    val = np.deg2rad(val)
                joint_values.append(val)
            api.goto_joint(*joint_values) # Unpack list into arguments
        except ValueError:
            print("Invalid input for Joint position.")
        except Exception as e:
             print(f"Error in Go To Joint: {e}")

    ttk.Button(joint_goto_frame, text="Go To Joints", command=_goto_joint_pos).grid(row=1, column=4, columnspan=4, pady=5)

    # === Joint Jogging Section ===
    joint_jog_frame = ttk.LabelFrame(iiwa_frame, text="Jog Joints (Degrees)", padding="10")
    joint_jog_frame.pack(fill="x", pady=10)

    jog_amount = 1.0 # Degrees
    for i, label in enumerate(joint_labels):
        ttk.Label(joint_jog_frame, text=label+":").grid(row=i, column=0, padx=5, pady=2, sticky='w')
        # Use partial or lambda to pass arguments correctly
        ttk.Button(joint_jog_frame, text="-", width=3, command=lambda l=label: adjust_joint(l, -jog_amount)).grid(row=i, column=1, padx=2, pady=2)
        ttk.Button(joint_jog_frame, text="+", width=3, command=lambda l=label: adjust_joint(l, jog_amount)).grid(row=i, column=2, padx=2, pady=2)

    # --- HandPoses Page ---
    hand_poses_page = ttk.Frame(notebook, padding="10")
    notebook.add(hand_poses_page, text="Hand Poses")

    # === Go to Position from HandPoses Section ===
    hand_poses_frame = ttk.LabelFrame(hand_poses_page, text="Go to Position from HandPoses", padding="10")
    hand_poses_frame.pack(fill="x", pady=10)

    hand_poses_entry_frame = ttk.Frame(hand_poses_frame)
    hand_poses_entry_frame.pack(pady=5)

    ttk.Label(hand_poses_entry_frame, text="Position Number:").pack(side=tk.LEFT, padx=5)
    hand_poses_entry = ttk.Entry(hand_poses_entry_frame, width=5)
    hand_poses_entry.pack(side=tk.LEFT, padx=5)
    hand_poses_entry.insert(0, "1")  # Default value

    hand_poses_buttons_frame = ttk.Frame(hand_poses_frame)
    hand_poses_buttons_frame.pack(pady=5)

    ttk.Button(hand_poses_buttons_frame, text="Go To Joints", 
               command=lambda: sequences.go_to_handpose_joints(-1+int(hand_poses_entry.get()))).pack(side=tk.LEFT, padx=10)
    ttk.Button(hand_poses_buttons_frame, text="Go To Position", 
               command=lambda: sequences.go_to_handpose_position(-1+int(hand_poses_entry.get()))).pack(side=tk.LEFT, padx=10)
    
    # === HandPoses Table Section ===
    hand_poses_table_frame = ttk.LabelFrame(hand_poses_page, text="Available Hand Poses", padding="10")
    hand_poses_table_frame.pack(fill="x", pady=10)

    def load_and_display_hand_poses():
        try:
            # Clear existing items in the table
            for item in hand_poses_tree.get_children():
                hand_poses_tree.delete(item)
                
            # Load the HandPoses.json file
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'communication', 'HandPoses.json')
            with open(json_path, 'r') as f:
                hand_poses = json.load(f)
                
            # Add each pose to the table
            for i, pose in enumerate(hand_poses):
                name = pose.get('name', f"Position {i+1}")
                hand_poses_tree.insert('', 'end', values=(i+1, name))
                
        except Exception as e:
            print(f"Error loading hand poses: {e}")

    # Create a treeview for the table
    hand_poses_tree = ttk.Treeview(hand_poses_table_frame, columns=('Index', 'Name'), show='headings', height=6)
    hand_poses_tree.heading('Index', text='Index')
    hand_poses_tree.heading('Name', text='Name')
    hand_poses_tree.column('Index', width=50, anchor='center')
    hand_poses_tree.column('Name', width=150)
    hand_poses_tree.pack(fill='both', expand=True)

    # Add a scrollbar
    scrollbar = ttk.Scrollbar(hand_poses_table_frame, orient="vertical", command=hand_poses_tree.yview)
    scrollbar.pack(side='right', fill='y')
    hand_poses_tree.configure(yscrollcommand=scrollbar.set)

    # Button to refresh the table
    ttk.Button(hand_poses_table_frame, text="Refresh Poses", command=load_and_display_hand_poses).pack(pady=5)

    # Load poses initially
    load_and_display_hand_poses()

    # Double-click to fill the position entry
    def on_pose_selected(event):
        selected_item = hand_poses_tree.selection()
        if selected_item:
            index = hand_poses_tree.item(selected_item[0], 'values')[0]
            hand_poses_entry.delete(0, tk.END)
            hand_poses_entry.insert(0, str(index))
            
    hand_poses_tree.bind('<Double-1>', on_pose_selected)

    # Add save current pose button to this page as well
    ttk.Separator(hand_poses_page, orient='horizontal').pack(fill='x', pady=15)
    ttk.Button(hand_poses_page, text="Save Current Pose to HandPoses.json",
               command=lambda: sequences.save_current_joints_to_file(camera_handler)).pack(pady=5, fill='x')



    # --- Gripper Page ---
    gripper_frame = ttk.Frame(notebook, padding="10")
    notebook.add(gripper_frame, text="Gripper & LED")
    grip_buttons_frame = ttk.Frame(gripper_frame)
    grip_buttons_frame.pack(pady=10)
    ttk.Button(grip_buttons_frame, text="Initialize Gripper", command=api.init_gripper).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(grip_buttons_frame, text="Open Gripper", command=api.open_gripper).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Force input frame
    force_frame = ttk.Frame(grip_buttons_frame)
    force_frame.pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Label(force_frame, text="Force:").pack(side=tk.LEFT)
    force_entry = ttk.Entry(force_frame, width=5)
    force_entry.pack(side=tk.LEFT, padx=2)
    force_entry.insert(0, "1")  # Default value
    
    # Close gripper with specified force
    ttk.Button(grip_buttons_frame, text="Close Gripper", 
               command=lambda: api.close_gripper(force=int(force_entry.get()))).pack(side=tk.LEFT, padx=5, pady=5)

    # --- ADD THIS LINE ---
    ttk.Button(grip_buttons_frame, text="Release Object", command=api.release_object).pack(side=tk.LEFT, padx=5, pady=5)
    # ------ END ADDITION ------

    # The Get Gripper State button is currently outside the frame, keep it there or move if desired
    ttk.Button(gripper_frame, text="Get Gripper State", command=api.get_gripper_state).pack(pady=5)

    ttk.Separator(gripper_frame, orient='horizontal').pack(fill='x', pady=15)

    # LED Control
    led_frame = ttk.Frame(gripper_frame)
    led_frame.pack(pady=10)
    ttk.Label(led_frame, text="Set LED Color:").pack(side=tk.LEFT, padx=10)
    ttk.Button(led_frame, text="Red", command=lambda: api.set_led("red")).pack(side=tk.LEFT, padx=5)
    ttk.Button(led_frame, text="Green", command=lambda: api.set_led("green")).pack(side=tk.LEFT, padx=5)
    ttk.Button(led_frame, text="Blue", command=lambda: api.set_led("blue")).pack(side=tk.LEFT, padx=5)

    # --- Sequences/Actions Page ---
    seq_frame = ttk.Frame(notebook, padding="10")
    notebook.add(seq_frame, text="Sequences & Actions")

    # Requires passing camera_handler to this function if sequences need it
    ttk.Button(seq_frame, text="Execute Full Sequence (Move+GoAround)",
               command=lambda: sequences.execute_sequence(camera_handler, clean_folder=True, locations=[6])).pack(pady=5, fill='x') # Add options later
    ttk.Button(seq_frame, text="Go Around Positions (Current Location)",
               command=lambda: sequences.go_around_positions(camera_handler)).pack(pady=5, fill='x')
    ttk.Button(seq_frame, text="Run Calibration Capture Routine",
               command=lambda: sequences.move_to_hand_poses_and_capture(camera_handler)).pack(pady=5, fill='x')
    ttk.Button(seq_frame, text="Run JustPickIt Sequence",
               command=lambda: sequences.just_pick_it_full_sequence()).pack(pady=5, fill='x')


    ttk.Separator(seq_frame, orient='horizontal').pack(fill='x', pady=15)

    ttk.Button(seq_frame, text="Save Current Pose to HandPoses.json",
               command=lambda: sequences.save_current_joints_to_file(camera_handler)).pack(pady=5, fill='x')
    # Add buttons for get pose, get joints etc. if needed for debugging
    get_state_frame = ttk.Frame(seq_frame)
    get_state_frame.pack(pady=10)
    ttk.Button(get_state_frame, text="Get KMR Pose", command=api.get_pose).pack(side=tk.LEFT, padx=5)
    ttk.Button(get_state_frame, text="Get IIWA Pos", command=api.get_iiwa_position).pack(side=tk.LEFT, padx=5)
    ttk.Button(get_state_frame, text="Get IIWA Joints", command=api.get_iiwa_joint_position).pack(side=tk.LEFT, padx=5)

    ttk.Button(get_state_frame, text="Go_to_the_position", command=sequences.Go_to_the_position).pack(side=tk.LEFT, padx=5)


    ttk.Separator(seq_frame, orient='horizontal').pack(fill='x', pady=15)

    # Zone Check
    zone_frame = ttk.Frame(seq_frame)
    zone_frame.pack(pady=10)

    ttk.Label(zone_frame, text="X:").grid(row=0, column=0)
    x_entry = ttk.Entry(zone_frame, width=5)
    x_entry.grid(row=0, column=1)
    x_entry.insert(0, "0.0")

    ttk.Label(zone_frame, text="Y:").grid(row=0, column=2)
    y_entry = ttk.Entry(zone_frame, width=5)
    y_entry.grid(row=0, column=3)
    y_entry.insert(0, "0.0")

    ttk.Label(zone_frame, text="Orientation:").grid(row=0, column=4)
    orientation_entry = ttk.Entry(zone_frame, width=5)
    orientation_entry.grid(row=0, column=5)
    orientation_entry.insert(0, "0.0")

    ttk.Label(zone_frame, text="Zone:").grid(row=0, column=6)
    zone_entry = ttk.Entry(zone_frame, width=3)
    zone_entry.grid(row=0, column=7)
    zone_entry.insert(0, "1")

    def check_zone():
        try:
            x = float(x_entry.get())
            y = float(y_entry.get())
            orientation = float(orientation_entry.get())
            zone = int(zone_entry.get())
            result = api.IsPositionInZone(x, y, orientation, zone)
            print(f"Position in Zone {zone}: {result}")
        except ValueError:
            print("Invalid input for zone check.")

    ttk.Button(zone_frame, text="Check Zone", command=check_zone).grid(row=0, column=8, padx=5)

    
    # --- Advanced Sequence Configuration Page ---
    advanced_seq_frame = ttk.Frame(notebook, padding="10")
    notebook.add(advanced_seq_frame, text="Advanced Sequence")
    
    # Create a frame for all the parameters
    params_frame = ttk.LabelFrame(advanced_seq_frame, text="Execute Sequence Parameters", padding="10")
    params_frame.pack(fill="x", pady=10)
    
    # Checkbox parameters
    only_current_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Only Current Location", variable=only_current_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
    
    do_camera_around_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Do Camera Around", variable=do_camera_around_var).grid(row=0, column=1, sticky="w", padx=5, pady=2)
    
    take_images_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Take Images", variable=take_images_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    do_detection_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Do Detection", variable=do_detection_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    do_6d_estimation_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Do 6D Estimation", variable=do_6d_estimation_var).grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    go_to_object_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(params_frame, text="Go To Object", variable=go_to_object_var).grid(row=2, column=1, sticky="w", padx=5, pady=2)
    
    clean_folder_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(params_frame, text="Clean Folder", variable=clean_folder_var).grid(row=3, column=0, sticky="w", padx=5, pady=2)
    
    # Text entry parameters
    entry_frame = ttk.Frame(params_frame)
    entry_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="w")
    
    ttk.Label(entry_frame, text="Detection Item:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    detection_item_entry = ttk.Entry(entry_frame, width=20)
    detection_item_entry.grid(row=0, column=1, padx=5, pady=2)
    detection_item_entry.insert(0, "bottle")  # Default value
    
    ttk.Label(entry_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
    output_folder_entry = ttk.Entry(entry_frame, width=40)
    output_folder_entry.grid(row=1, column=1, padx=5, pady=2)
    output_folder_entry.insert(0, config.DEFAULT_GO_AROUND_OUTPUT_FOLDER)  # Default value
    
    # Execute button
    def run_advanced_sequence():
        try:
            sequences.execute_sequence(
                camera_handler,
                Only_current=only_current_var.get(),
                do_camera_around=do_camera_around_var.get(),
                take_images=take_images_var.get(),
                do_detection=do_detection_var.get(),
                do_6d_estimation=do_6d_estimation_var.get(),
                go_to_object=go_to_object_var.get(),
                detection_item=detection_item_entry.get(),
                clean_folder=clean_folder_var.get(),
                output_folder=output_folder_entry.get()
            )
        except Exception as e:
            print(f"Error executing advanced sequence: {e}")
    
    ttk.Button(advanced_seq_frame, text="Execute Sequence", 
               command=run_advanced_sequence).pack(pady=15, fill="x")
    # Add Honk buttons if desired
    # ttk.Button(seq_frame, text="Honk On", command=api.honk_on).pack(pady=5)
    # ttk.Button(seq_frame, text="Honk Off", command=api.honk_off).pack(pady=5)

    


    root.mainloop()