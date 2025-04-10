import tkinter as tk
import tkinter.ttk as ttk
import numpy as np

# --- Use Relative Imports ---
from . import config
from . import kuka_api as api
from . import sequences # If GUI calls sequence functions


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

    # --- Gripper Page ---
    gripper_frame = ttk.Frame(notebook, padding="10")
    notebook.add(gripper_frame, text="Gripper & LED")

    grip_buttons_frame = ttk.Frame(gripper_frame)
    grip_buttons_frame.pack(pady=10)
    ttk.Button(grip_buttons_frame, text="Initialize Gripper", command=api.init_gripper).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(grip_buttons_frame, text="Open Gripper", command=api.open_gripper).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Button(grip_buttons_frame, text="Close Gripper", command=lambda: api.close_gripper(force=1)).pack(side=tk.LEFT, padx=5, pady=5)

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
               command=lambda: sequences.just_pick_it_full_sequence(camera_handler)).pack(pady=5, fill='x')


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


    # Add Honk buttons if desired
    # ttk.Button(seq_frame, text="Honk On", command=api.honk_on).pack(pady=5)
    # ttk.Button(seq_frame, text="Honk Off", command=api.honk_off).pack(pady=5)

    


    root.mainloop()