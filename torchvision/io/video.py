import av
import torch
import numpy as np
import math


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


def _read_from_stream(container, start_offset, end_offset, stream, stream_name):
    container.seek(start_offset, any_frame=False, backward=True, stream=stream)
    frames = []
    first_frame = None
    for idx, frame in enumerate(container.decode(**stream_name)):
        if frame.pts < start_offset:
            first_frame = frame
            continue
        if first_frame and first_frame.pts < start_offset:
            audio_frames.append(first_frame)
            first_frame = None
        frames.append(frame)
        if frame.pts > end_offset:
            break
    return frames


def read_video(filename, start_pts=0, end_pts=math.inf):
    container = av.open(filename)

    video_frames = []
    if container.streams.video:
        video_frames = _read_from_stream(container, start_pts, end_pts, container.streams.video[0], {'video': 0})
    audio_frames = []
    if container.streams.audio:
        audio_frames = _read_from_stream(container, start_pts, end_pts, container.streams.audio[0], {'audio': 0})

    container.close()

    vframes = [frame.to_rgb().to_ndarray() for frame in video_frames]
    aframes = [frame.to_ndarray() for frame in audio_frames]
    vframes = torch.as_tensor(np.stack(vframes))
    if aframes:
        aframes = np.concatenate(aframes, 1)
        aframes = torch.as_tensor(aframes)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    # return video_frames, audio_frames
    return vframes, aframes


def _read_video(filename, start_offset, end_offset):
    container = av.open(filename)

    # video
    container.seek(start_offset, any_frame=False, backward=True, stream=container.streams.video[0])
    video_frames = []
    for idx, frame in enumerate(container.decode(video=0)):
        if frame.pts < start_offset:
            continue
        if frame.pts > end_offset:
            break
        video_frames.append(frame)

    # audio
    container.seek(start_offset, backward=True, any_frame=False, stream=container.streams.audio[0])
    audio_frames = []
    first_frame = None
    for idx, frame in enumerate(container.decode(audio=0)):
        if frame.pts < start_offset:
            first_frame = frame
            continue
        if first_frame and first_frame.pts < start_offset:
            audio_frames.append(first_frame)
            first_frame = None
        audio_frames.append(frame)
        if frame.pts > end_offset:
            break

    container.close()
    return video_frames, audio_frames
