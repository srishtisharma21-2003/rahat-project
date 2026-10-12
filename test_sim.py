# test_sim.py
from simulation.robot_sim import RAHATSimulation
import cv2
import os

print("Initializing simulation...")
sim = RAHATSimulation()

print("Moving robot and capturing frame...")
# Target position: slightly above the tissue, at the center
target_position = [0.4, 0.0, 0.05]

# Step the simulation
frame = sim.step(target_position)

# Save the image so we can look at it
# We use imwrite instead of imshow because Kali Linux might be running headless
# or without a display server, and imshow can crash.
output_path = os.path.expanduser("~/rahat-project/test_ultrasound.jpg")
cv2.imwrite(output_path, frame)

print(f"Success! Ultrasound frame saved to: {output_path}")

# Get joint states to verify the robot moved
joints = sim.get_joint_states()
print(f"Robot joint 0 angle: {joints[0]['position']:.4f} radians")
print(f"Robot end-effector position: {sim.get_end_effector_position()}")

sim.close()
print("Simulation closed cleanly.")
