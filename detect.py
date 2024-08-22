import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import ClientSession

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
class VideoStreamTrack(VideoStreamTrack):
    def __init__(self, track, model, device):
        super().__init__()
        self.track = track
        self.model = model
        self.device = device
        self.person_detected_start_time = None

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        return img
        img_resized = cv2.resize(img, (320, 320))  # Resize for YOLOv5

        # Convert image to tensor and prepare for YOLOv5
        img_tensor = (
            torch.from_numpy(img_resized)
            .to(
                self.device, non_blocking=True
            )  # Transfer tensor to device with non-blocking
            .permute(2, 0, 1)  # Change dimension order [HWC] -> [CHW]
            .unsqueeze(0)  # Add batch dimension [1, C, H, W]
        ).float() / 255.0  # Normalize to [0, 1]

        # Use FP16 if available for faster inference
        if self.device.type != "cpu":
            img_tensor = img_tensor.half()

        # YOLOv5 inference
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = non_max_suppression(pred)

        person_detected = False

        # Annotate frame
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(
                    img_tensor.shape[2:], det[:, :4], img.shape
                ).round()
                annotator = Annotator(img, line_width=2, example=str(self.model.names))
                for *xyxy, conf, cls in reversed(det):
                    label = f"{self.model.names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    if self.model.names[int(cls)] == "person":
                        person_detected = True

                img = annotator.result()

        # Handle detection timing
        if person_detected:
            if self.person_detected_start_time is None:
                self.person_detected_start_time = time.time()
            elif time.time() - self.person_detected_start_time >= 5:
                print("person detected")
        else:
            self.person_detected_start_time = None
        return img  # Return annotated frame


async def negotiate(pc, url):
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    async with ClientSession() as session:
        async with session.post(
            f"{url}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as response:
            answer = await response.json()
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )


async def vidstream():
    url = "http://127.0.0.1:3123"  # Ganti dengan URL server Anda
    pc = RTCPeerConnection()
    device = select_device("cpu")
    model = DetectMultiBackend("yolov5s.pt", device=device)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print("Video track received")
            video_track = VideoStreamTrack(track, model, device)
            while True:
                frame = await video_track.recv()
                cv2.imshow("YOLOv5 Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    await negotiate(pc, url)

    while True:
        await asyncio.sleep(1)

    await pc.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # opt = parse_opt()
    asyncio.run(vidstream())
    # main(opt)
