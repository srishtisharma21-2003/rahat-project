import asyncio
import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai.unet_model import TinyUNet
from simulation.robot_sim import RAHATSimulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing simulation and AI model...")
    model = TinyUNet()
    weights_path = PROJECT_ROOT / "data" / "weights.pth"
    if weights_path.is_file():
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        print(f"Loaded trained AI weights from {weights_path}")
    else:
        print(f"No trained weights found at {weights_path}; using random weights.")
    model.eval()
    sim = RAHATSimulation()

    app.state.model = model
    app.state.sim = sim
    app.state.simulation_lock = asyncio.Lock()

    try:
        yield
    finally:
        sim.close()


app = FastAPI(title="RAHAT Simulation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "running", "project": "RAHAT Simulation"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Stream simulation frames and model predictions to a connected client."""
    await websocket.accept()
    print("Client connected")

    sim = websocket.app.state.sim
    model = websocket.app.state.model
    simulation_lock = websocket.app.state.simulation_lock

    num_steps = 100
    angles = np.linspace(0, 2 * np.pi, num_steps)

    try:
        for step_idx, angle in enumerate(angles):
            target_x = 0.4 + 0.08 * np.cos(angle)
            target_y = 0.08 * np.sin(angle)
            target_z = 0.05
            scan_position = [float(target_x), float(target_y), target_z]

            async with simulation_lock:
                frame = sim.step(scan_position)
                joint_states = sim.get_joint_states()
                ee_position = sim.get_end_effector_position()

            frame_tensor = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            )

            with torch.inference_mode():
                mask = model(frame_tensor)

            mask_np = mask.squeeze().cpu().numpy()
            confidence = float(mask_np.mean())
            hemorrhage_detected = confidence > 0.3

            frame_encoded, jpeg_buffer = cv2.imencode(".jpg", frame)
            if not frame_encoded:
                raise RuntimeError("Failed to encode ultrasound frame as JPEG")
            frame_b64 = base64.b64encode(jpeg_buffer).decode("utf-8")

            mask_uint8 = (mask_np * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
            mask_encoded, mask_buffer = cv2.imencode(".png", mask_colored)
            if not mask_encoded:
                raise RuntimeError("Failed to encode AI mask as PNG")
            mask_b64 = base64.b64encode(mask_buffer).decode("utf-8")

            await websocket.send_json({
                "step": step_idx,
                "total_steps": num_steps,
                "joint_states": joint_states,
                "end_effector_position": ee_position,
                "ultrasound_frame": frame_b64,
                "ai_mask": mask_b64,
                "hemorrhage_detected": hemorrhage_detected,
                "confidence": confidence,
                "scan_position": scan_position,
            })

            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("Client disconnected")
    else:
        print("Scan complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
