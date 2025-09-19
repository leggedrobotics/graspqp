import bpy
import os
import glob
import math
import mathutils

HAND_TYPES = ["shadow_hand", "ability_hand", "robotiq3", "robotiq2", "allegro"]
GRASP_TYPES = ["default", "pinch", "precision"]
ENERGY_METHODS = ["span_overall_cone_sqp_default_longer_gendex"]

for hand_type in HAND_TYPES:
    for grasp_type in GRASP_TYPES:
        for energy_method in ENERGY_METHODS:
            

            # hand_type = "shadow_hand"
            # grasp_type = "default"
            n_contacts = "12_contacts"
            # energy_method = "span_overall_cone_sqp_default_longer_gendex" #"span_overall_cone_qp"




            # Path to your folder containing OBJ files
            folder_path = "/home/zrene/git/DexGraspNet/graspqp/_vis/interaction_meshes"
            screenshot_path = f"/home/zrene/git/DexGraspNet/graspqp/_vis/interaction_meshes/{hand_type}/{n_contacts}/{energy_method}/{grasp_type}"
            os.makedirs(screenshot_path, exist_ok=True)
            max_assets = 1e6

            # Spacing between the objects
            spacing = 0.4

            # Collect all .obj files in the folder
            obj_files = glob.glob(os.path.join(folder_path, "*", hand_type, n_contacts, energy_method, grasp_type, "*.obj"), recursive=True)
            obj_files.sort()  # Sort files for consistent placement order

            # Clear the scene (optional)
            def clear_scene():
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete(use_global=False)

            clear_scene()

            # Create or get "material_0"
            #if "material_0" not in bpy.data.materials:
            #    mat = bpy.data.materials.new(name="VertexColor")
            #else:
            mat = bpy.data.materials["VertexColor"]



            # Add a large ground plane at height zero
            bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
            ground_plane = bpy.context.active_object
            ground_plane.scale = (100, 100, 1)


            # Create or get "material_0"
            if "planemat" not in bpy.data.materials:
                planemat = bpy.data.materials.new(name="planemat")
            else:
                planemat = bpy.data.materials["planemat"]
            # Optional: assign material_0 to the ground plane too
            if ground_plane.type == 'MESH':
                if len(ground_plane.data.materials) == 0:
                    ground_plane.data.materials.append(planemat)
                else:
                    ground_plane.data.materials[0] = planemat
                    
                    
                    

            obj_idx = 0

            n_assets = min(max_assets, len(obj_files))
            grid_length = int(math.sqrt(n_assets))

            # Load and place each OBJ file
            for i, file_path in enumerate(obj_files):
                # Import the OBJ file
                bpy.ops.wm.obj_import(filepath=file_path)

                # Get the newly imported objects (they are automatically selected)
                imported_objects = bpy.context.selected_objects

                # Offset in X for layout
                offset_x = (obj_idx // grid_length) * spacing
                offset_y = (obj_idx % grid_length) * spacing

                for obj in imported_objects:
                    obj_idx += 1
                    obj.rotation_euler[0] -= math.radians(90)
                    # Move the object
                    obj.location.x += offset_x
                    obj.location.y += offset_y
                    
                # Update scene to get bounding box data
                    bpy.context.view_layer.update()

                    # Align base of object to Z=0
                    bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
                    z_min = min([v.z for v in bbox_world])
                    obj.location.z += -z_min

                    # Assign material_0
                    if obj.type == 'MESH':
                        if len(obj.data.materials) == 0:
                            obj.data.materials.append(mat)
                        else:
                            obj.data.materials[0] = mat

                if i > max_assets:
                    break

            print(f"Imported {len(obj_files[:4])} OBJ files and applied 'material_0' to them.")

            # Delete existing lights (optional cleanup)
            for light in [obj for obj in bpy.data.objects if obj.type == 'LIGHT']:
                bpy.data.objects.remove(light, do_unlink=True)

            # Add a Sun Lamp
            bpy.ops.object.light_add(type='SUN', radius=1, location=(5, -5, 10))
            sun = bpy.context.object
            sun.data.energy = 5
            sun.rotation_euler = (math.radians(60), math.radians(0), math.radians(45))

            # Add an Area Light for soft fill
            bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
            area = bpy.context.object
            area.data.energy = 300
            area.data.size = 5
            area.rotation_euler = (math.radians(-45), math.radians(0), math.radians(45))

            # Optional: Use Cycles for better lighting results (comment out if using Eevee)
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.device = 'GPU'  # Set to 'CPU' if no GPU

            center_point = (grid_length * spacing / 2 , grid_length * spacing / 2, 0.05)


            # Create a camera
            # Compute the camera look-at target point
            target = mathutils.Vector((grid_length * spacing / 2, grid_length * spacing / 2, 0.05))

            # Add a camera at (0, 0, 1)
            bpy.ops.object.camera_add(location=(grid_length * spacing * 2.5 , grid_length * spacing / 2, 2.5))
            camera = bpy.context.object

            # Compute direction to target
            direction = target - camera.location
            rot_quat = direction.to_track_quat('-Z', 'Y')  # Look down -Z, up is Y

            # Apply rotation to camera
            camera.rotation_euler = rot_quat.to_euler()

            # Set camera as active
            bpy.context.scene.camera = camera


            # Set the output file path (absolute or relative to the .blend file)
            bpy.context.scene.render.filepath = os.path.join(screenshot_path, "render.png")

            # Set the image format to PNG
            bpy.context.scene.render.image_settings.file_format = 'PNG'

            # Render the scene and save the image
            bpy.ops.render.render(write_still=True)