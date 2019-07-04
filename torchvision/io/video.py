import gc
import torch
import numpy as np

try:
    import av
except ImportError:
    av = None


def _check_av_available():
    if av is None:
        raise ImportError("""\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")


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
    _check_av_available()
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
            if frame.pts != start_offset:
                frames.append(first_frame)
            first_frame = None
        frames.append(frame)
        if frame.pts >= end_offset:
            break
    return frames


def _align_audio_frames(aframes, audio_frames, ref_start, ref_end):
    start, end = audio_frames[0].pts, audio_frames[-1].pts
    total_aframes = aframes.shape[1]
    step_per_aframe = (end - start + 1) / total_aframes
    s_idx = 0
    e_idx = total_aframes
    if start < ref_start:
        s_idx = int((ref_start - start) / step_per_aframe)
    if end > ref_end:
        e_idx = int((ref_end - end) / step_per_aframe)
    return aframes[:, s_idx:e_idx]


def read_video(filename, start_pts=0, end_pts=None):
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
        info (Dict): metadata for the video and audio. Can contain the fields
            - video_fps (float)
            - audio_fps (int)
    """
    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))

    container = av.open(filename)
    info = {}

    video_frames = []
    if container.streams.video:
        video_frames = _read_from_stream(container, start_pts, end_pts,
                                         container.streams.video[0], {'video': 0})
        info["video_fps"] = float(container.streams.video[0].average_rate)
    audio_frames = []
    if container.streams.audio:
        audio_frames = _read_from_stream(container, start_pts, end_pts,
                                         container.streams.audio[0], {'audio': 0})
        info["audio_fps"] = container.streams.audio[0].rate

    container.close()

    vframes = [frame.to_rgb().to_ndarray() for frame in video_frames]
    aframes = [frame.to_ndarray() for frame in audio_frames]
    vframes = torch.as_tensor(np.stack(vframes))
    if aframes:
        aframes = np.concatenate(aframes, 1)
        aframes = torch.as_tensor(aframes)
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    return vframes, aframes, info


def read_video_timestamps(filename):
    """
    List the video frames timestamps.

    Note that the function decodes the whole video frame-by-frame.

    Arguments:
        filename (str): path to the video file

    Returns:
        pts (List[int]): presentation timestamps for each one of the frames
            in the video.
    """
    _check_av_available()
    container = av.open(filename)

    video_frames = []
    if container.streams.video:
        video_frames = _read_from_stream(container, 0, float("inf"),
                                         container.streams.video[0], {'video': 0})
    container.close()
    return [x.pts for x in video_frames]
