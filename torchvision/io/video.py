import av
import torch


def write_video(filename, video_array, fps):
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    container = av.open(filename, mode='w')

    stream = container.add_stream('mpeg4', rate=fps)
    stream.width = video_array.shape[2]
    stream.height = video_array.shape[1]
    stream.pix_fmt = 'yuv420p'

    for img in video_array:
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()
