# simulation/robot_sim.py
import pybullet as p
import pybullet_data
import numpy as np
import cv2


class RAHATSimulation:
    def __init__(self):
        # 1. Connect to PyBullet in DIRECT mode (Headless - no graphics window)
        # This is critical for your i3 processor to prevent lag.
        self.physics_client = p.connect(p.DIRECT)

        # 2. Tell PyBullet where to find built-in robot models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 3. Set gravity (pointing down, like real life)
        p.setGravity(0, 0, -9.81)

        # 4. Load the KUKA iiwa robot arm
        # useFixedBase=True means the base of the robot stays in one place
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # 5. Identify the end-effector (the "hand" of the robot)
        # The KUKA iiwa has 7 joints (indices 0 to 6). Index 6 is the tip.
        self.end_effector_index = 6

        # 6. Create the tissue surface (a flat box)
        # halfExtents defines the size of the box (15cm x 15cm x 2cm)
        tissue_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.02])
        # Base position is at x=0.4 (in front of the robot), y=0, z=0
        self.tissue_body = p.createMultiBody(0, tissue_shape, basePosition=[0.4, 0, 0])

        # 7. Create the hemorrhage (hidden inside the tissue)
        # A small sphere with a radius of 1.5cm
        hemorrhage_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.015)
        # Positioned inside the tissue box (z=0.01 is just below the surface)
        self.hemorrhage_body = p.createMultiBody(
            0, hemorrhage_shape,
            basePosition=[0.4, 0.03, 0.01]
        )

        # Store the number of joints for later use
        self.num_joints = p.getNumJoints(self.robot_id)

    def move_probe_to(self, target_position):
        """
        Move the robot's end-effector to a target XYZ position using Inverse Kinematics.
        """
        # calculateInverseKinematics takes the robot ID, the end-effector link,
        # and the target [x, y, z] position. It returns the required joint angles.
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position
        )

        # Apply the computed joint angles to each joint using position control
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i]
            )

        return joint_poses

    def get_joint_states(self):
        """Get current joint angles and velocities (useful for the frontend later)."""
        states = []
        for i in range(self.num_joints):
            state = p.getJointState(self.robot_id, i)
            states.append({
                "joint": i,
                "position": state[0],  # angle in radians
                "velocity": state[1]
            })
        return states

    def get_end_effector_position(self):
        """Get the current XYZ position of the robot's tip."""
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        return list(link_state[4])  # [x, y, z] of the link frame

    def capture_ultrasound_frame(self, width=224, height=224):
        """
        Render a simulated ultrasound image using a depth camera.
        """
        # 1. Define camera position: above the tissue, looking down
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.4, 0, 0.3],      # Camera at x=0.4, y=0, z=0.3
            cameraTargetPosition=[0.4, 0, 0],      # Looking at the tissue center
            cameraUpVector=[0, 1, 0]               # "Up" direction
        )

        # 2. Define camera lens properties (Field of View, aspect ratio, near/far clipping)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=1.0
        )

        # 3. Render the scene.
        # ER_TINY_RENDERER is a software renderer that works without a GPU (perfect for i3).
        # It returns: width, height, RGB array, Depth array, Segmentation mask
        _, _, rgb, depth, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        # 4. Convert depth buffer to a numpy array
        depth_array = np.array(depth).reshape(height, width)

        # 5. Normalize depth values to 0-255 (for image display)
        # Deep tissue will be dark, surface will be bright
        depth_normalized = (depth_array - depth_array.min()) / \
                          (depth_array.max() - depth_array.min() + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)

        # 6. Apply a "bone" colormap to make it look like a real ultrasound
        ultrasound = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_BONE)

        return ultrasound

    def step(self, target_position):
        """
        One simulation step:
        1. Move the robot to the target position
        2. Advance physics by one timestep
        3. Capture an ultrasound frame
        """
        self.move_probe_to(target_position)
        p.stepSimulation()  # Advance physics by 1/240th of a second
        frame = self.capture_ultrasound_frame()
        return frame

    def close(self):
        """Disconnect from the physics server."""
        p.disconnect()
