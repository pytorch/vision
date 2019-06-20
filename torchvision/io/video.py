import av
import gc
import torch
import numpy as np
import math


# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 20


def write_video(filename, video_array, fps):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Arguments:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): frames per second
    """
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
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

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
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames

    Arguments:
        filename (str): path to the video file
        start_pts (int, optional): the start presentation time of the video
        end_pts (int, optional): the end presentation time

    Returns:
        vframes (Tensor[T, H, W, C]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
    """
    container = av.open(filename)

    video_frames = []
    if container.streams.video:
        video_frames = _read_from_stream(container, start_pts, end_pts,
                                         container.streams.video[0], {'video': 0})
    audio_frames = []
    if container.streams.audio:
        audio_frames = _read_from_stream(container, start_pts, end_pts,
                                         container.streams.audio[0], {'audio': 0})

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


def read_video_timestamps(filename):
    container = av.open(filename)

    video_frames = []
    if container.streams.video:
        video_frames = _read_from_stream(container, 0, math.inf,
                                         container.streams.video[0], {'video': 0})
    container.close()
    return [x.pts for x in video_frames]
