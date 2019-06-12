import av
import gc
import warnings



_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 20


# remove warnings
av.logging.set_level(av.logging.ERROR)



class VideoReader(object):
    """
    Simple wrapper around PyAV that exposes a few useful functions for
    dealing with video reading.
    """
    def __init__(self, video_path, sampling_rate=1, decode_lossy=False, audio_resample_rate=None):
        """
        Arguments:
            video_path (str): path of the video to be loaded
        """
        self.container = av.open(video_path)
        self.sampling_rate = sampling_rate
        self.resampler = None
        if audio_resample_rate is not None:
            self.resampler = av.AudioResampler(rate=audio_resample_rate)
            
        
        if self.container.streams.video:
            # enable multi-threaded video decoding
            if decode_lossy:
                warnings.warn('VideoReader| thread_type==AUTO can yield potential frame dropping!', RuntimeWarning)
                self.container.streams.video[0].thread_type = 'AUTO'
            self.video_stream = self.container.streams.video[0]
        else:
            self.video_stream = None
 
    def seek(self, offset, backward=True, any_frame=False):
        stream = self.video_stream
        self.container.seek(offset, any_frame=any_frame, backward=backward, stream=stream)

    def _occasional_gc(self):
        # there are a lot of reference cycles in PyAV, so need to manually call
        # the garbage collector from time to time
        global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
            gc.collect()

    def _read_video(self, offset, num_frames):
        self._occasional_gc()
        self.seek(offset)
        video_frames = []
        count = 0
        for idx, frame in enumerate(self.container.decode(video=0)):
            if frame.pts < offset:
                continue
            video_frames.append(frame)
            if count >= num_frames - 1:
                break
            count += 1
        return video_frames
    
    def _resample_audio_frame(self, frame):
        curr_pts = frame.pts
        frame.pts = None
        frame = self.resampler.resample(frame)
        frame.pts = curr_pts
        return frame


    def _read_audio(self, offset, end_offset):
        self._occasional_gc()
        if not self.container.streams.audio:
            return []

        self.container.seek(offset, backward=True, any_frame=False, stream=self.container.streams.audio[0])

        audio_frames = []
        first_frame = None
        for idx, frame in enumerate(self.container.decode(audio=0)):
            if frame.pts < offset:
                first_frame = frame
                continue
            if first_frame and first_frame.pts < offset:
                if self.resampler is not None:
                    first_frame = self._resample_audio_frame(first_frame)
                audio_frames.append(first_frame)
                first_frame = None
            # if we want to resample audio to a different framerate 
            if self.resampler is not None:
                frame = self._resample_audio_frame(frame)
            audio_frames.append(frame)
            if frame.pts > end_offset:
                break
        return audio_frames

    def read(self, offset, num_frames):
        """
        Reads video frames and audio frames starting from offset.
        The number of video frames read is given by num_frames.
        The number of audio frames read is defined by the start and
        end time of the first and last video frames, respectively
        Arguments:
            offset (int): the start time from the read
            num_frames (int): the number of video frames to be read
        Returns:
            video_frames (List[av.VideoFrame])
            audio_frames (List[av.AudioFrame])
        """
        if self.container is None:
            return [], []

        num_frames = self.sampling_rate * num_frames
        video_frames = self._read_video(offset, num_frames)
        if len(video_frames) < 1:
            end_offset = offset
        elif len(video_frames) < 2:
            end_offset = video_frames[-1].pts
        else:
            step = video_frames[-1].pts - video_frames[-2].pts
            end_offset = video_frames[-1].pts + step - 1
        try:
            audio_frames = self._read_audio(offset, end_offset)
        except av.AVError:
            audio_frames = []
        return video_frames, audio_frames

    def list_keyframes(self):
        """
        Returns a list of start times for all the keyframes in the video
        Returns:
            keyframes (List[int])
        """
        keyframes = []
        if self.video_stream is None:
            return []
        pts = -1
        while True:
            try:
                self.seek(pts + 1, backward=False)
            except av.AVError:
                break
            packet = next(self.container.demux(video=0))
            pts = packet.pts
            #TODO: double check if this is needed
            if pts is None:
                # should we simply return []?
                return keyframes

            if packet.is_keyframe:
                keyframes.append(pts)
        return keyframes

    def _compute_end_video_pts(self):
        self.seek(self.container.duration, any_frame=True)
        end_step = next(self.container.demux(video=0)).pts
        if end_step is None:
            self.seek(self.container.duration, any_frame=False)
            gen = self.container.demux(video=0)
            last_pts = 0
            while True:
                last_pts = next(gen).pts
                if last_pts is None:
                    break
                end_step = last_pts
        return end_step

    def _compute_start_video_pts(self):
        self.seek(0)
        start = next(self.container.demux(video=0)).pts
        return start

    def _compute_step_pts(self):
        self.seek(0)
        pts = []
        num = 11
        gen = self.container.demux(video=0)
        for _ in range(num):
            next(gen)
        for _ in range(num):
            pts.append(next(gen).pts)
        print(pts)
        steps = [p1 - p2 for p1, p2 in zip(pts[1:], pts[:-1])]
        print(steps)
        steps = max(set(steps), key=steps.count)
        return int(steps)

    def _compute_step_pts(self):
        frames = self._read_video(0, 2)
        steps = frames[1].pts - frames[0].pts
        return steps

    def list_every(self, n_frames):
        step = 1 / float(self.video_stream.average_rate * self.video_stream.time_base)
        end = self._compute_end_video_pts()
        start = self._compute_start_video_pts()
        step = self._compute_step_pts()
        """
        orig_step = int(step)
        for i in range(-10, 10):
            if (end - start) % (orig_step + i) == 0:
                step = orig_step + i
                break
        """
        return list(range(start, end + 1, int(step)))[::n_frames]


    def _decode_every(self):
        """
        A function used for truly decoding every single frame. 
        This should not be used outside of the dataset indexing step
        Returns:
            timestamp of every frame within the video (List[int])
        """
        if self.video_stream is None or self.container is None:
            return []
        self.seek(0, backward=False)
        d = [p for p in self.container.decode(video=0)]
        return [x.pts for x in d[::1]]
