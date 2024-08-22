import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import ClientSession

async def negotiate(pc, url):
    # Add video and audio transceivers
    pc.addTransceiver('video', direction='recvonly')
    pc.addTransceiver('audio', direction='recvonly')

    # Create and set local offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send offer to server
    async with ClientSession() as session:
        async with session.post(f'{url}/offer', json={
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }) as response:
            answer = await response.json()

    # Set remote description
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

class VideoStreamTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        return frame

async def main():
    url = 'http://localhost:3123'  # Replace with your server URL

    pc = RTCPeerConnection()

    # Handle incoming tracks
    @pc.on('track')
    async def on_track(track):
        if track.kind == 'video':
            print("Video track received")
            while True:
                frame = await track.recv()
                frame_data = frame.to_ndarray(format='bgr24')
                cv2.imshow('Video', frame_data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    await negotiate(pc, url)

    # Keep running until 'q' is pressed
    while True:
        await asyncio.sleep(1)

    await pc.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
