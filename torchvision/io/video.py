import re
import gc
import torch
import numpy as np

try:
    import av
    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, 'pict_type'):
        av = ImportError("""\
Your version of PyAV is too old for the necessary video operations in torchvision.
If you are on Python 3.5, you will have to build from source (the conda-forge
packages are not up-to-date).  See
https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")
except ImportError:
    av = ImportError("""\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
""")


def _check_av_available():
    if isinstance(av, Exception):
        raise av


def _av_available():
    return not isinstance(av, Exception)


# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10


def write_video(filename, video_array, fps, video_codec='libx264', options=None):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Parameters
    ----------
    filename : str
        path where the video will be saved
    video_array : Tensor[T, H, W, C]
        tensor containing the individual frames, as a uint8 tensor in [T, H, W, C] format
    fps : Number
        frames per second
    """
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    container = av.open(filename, mode='w')

    stream = container.add_stream(video_codec, rate=fps)
    stream.width = video_array.shape[2]
    stream.height = video_array.shape[1]
    stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
    stream.options = options or {}

    for img in video_array:
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame.pict_type = 'NONE'
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

    frames = {}
    should_buffer = False
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(br"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(br"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    for idx, frame in enumerate(container.decode(**stream_name)):
        frames[frame.pts] = frame
        if frame.pts >= end_offset:
            if should_buffer and buffer_count < max_buffer_size:
                buffer_count += 1
                continue
            break
    # ensure that the results are sorted wrt the pts
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        first_frame_pts = max(i for i in frames if i < start_offset)
        result.insert(0, frames[first_frame_pts])
    return result


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

    Parameters
    ----------
    filename : str
        path to the video file
    start_pts : int, optional
        the start presentation time of the video
    end_pts : int, optional
        the end presentation time

    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[K, L]
        the audio frames, where `K` is the number of channels and `L` is the
        number of points
    info : Dict
        metadata for the video and audio. Can contain the fields video_fps (float)
        and audio_fps (int)
    """
    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))

    container = av.open(filename, metadata_errors='ignore')
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


def _can_read_timestamps_from_packets(container):
    extradata = container.streams[0].codec_context.extradata
    if extradata is None:
        return False
    if b"Lavc" in extradata:
        return True
    return False


def read_video_timestamps(filename):
    """
    List the video frames timestamps.

    Note that the function decodes the whole video frame-by-frame.

    Parameters
    ----------
    filename : str
        path to the video file

    Returns
    -------
    pts : List[int]
        presentation timestamps for each one of the frames in the video.
    video_fps : int
        the frame rate for the video

    """
    _check_av_available()
    container = av.open(filename, metadata_errors='ignore')

    video_frames = []
    video_fps = None
    if container.streams.video:
        if _can_read_timestamps_from_packets(container):
            # fast path
            video_frames = [x for x in container.demux(video=0) if x.pts is not None]
        else:
            video_frames = _read_from_stream(container, 0, float("inf"),
                                             container.streams.video[0], {'video': 0})
        video_fps = float(container.streams.video[0].average_rate)
    container.close()
    return [x.pts for x in video_frames], video_fps
