import os

import pytest
import torch
from torchvision.io import _HAS_VIDEO_DECODER, VideoReader

try:
    import av
except ImportError:
    av = None

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos")

test_videos = [
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi",
    "TrumanShow_wave_f_nm_np1_fr_med_26.avi",
    "v_SoccerJuggling_g23_c01.avi",
    "v_SoccerJuggling_g24_c01.avi",
    "R6llTwEh07w.mp4",
    "SOX5yA1l24A.mp4",
    "WUzgd7C1pWA.mp4",
]


def _yuv420_to_444(mat):
    # logic taken from
    # https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion
    width = mat.shape[-1]
    height = mat.shape[0] * 2 // 3
    luma = mat[:height]
    uv = mat[height:].reshape(2, height // 2, width // 2)
    uv2 = torch.nn.functional.interpolate(uv[None], scale_factor=2, mode='nearest')[0]
    yuv2 = torch.cat([luma[None], uv2]).permute(1, 2, 0)
    return yuv2


def _yuv420_to_rgb(mat, limited_color_range=True, standard='bt709'):
    # taken from https://en.wikipedia.org/wiki/YCbCr
    if standard == 'bt601':
        # ITU-R BT.601, as used by decord
        # taken from https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
        m = torch.tensor([[ 1.0000,  0.0000,  1.402],
                          [ 1.0000, -(1.772 * 0.114 / 0.587), -(1.402 * 0.299 / 0.587)],
                          [ 1.0000,  1.772,  0.0000]], device=mat.device)
    elif standard == 'bt709':
        # ITU-R BT.709
        # taken from https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
        m = torch.tensor([[ 1.0000,  0.0000,  1.5748],
                          [ 1.0000, -0.1873, -0.4681],
                          [ 1.0000,  1.8556,  0.0000]], device=mat.device)
    else:
        raise ValueError(f"{standard} not supported")

    if limited_color_range:
        # also present in https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
        # being mentioned as compensation for the footroom and headroom
        m = m * torch.tensor([255 / 219, 255 / 224, 255 / 224], device=mat.device)

    m = m.T

    # TODO: maybe this needs to come together with limited_color_range
    offset = torch.tensor([16., 128., 128.], device=mat.device)

    yuv2 = _yuv420_to_444(mat)

    res = (yuv2 - offset) @ m
    return res


@pytest.mark.skipif(_HAS_VIDEO_DECODER is False, reason="Didn't compile with support for gpu decoder")
class TestVideoGPUDecoder:
    @pytest.mark.skipif(av is None, reason="PyAV unavailable")
    def test_frame_reading(self):
        for test_video in test_videos:
            full_path = os.path.join(VIDEO_DIR, test_video)
            decoder = VideoReader(full_path, device="cuda:0")
            print(test_video)
            with av.open(full_path) as container:
                for av_frame in container.decode(container.streams.video[0]):
                    #print(av_frame.format)
                    av2 = av_frame.to_rgb().to_ndarray()
                    #print(av2.shape)
                    av_frames_yuv = torch.tensor(av_frame.to_ndarray())
                    #av_frames = torch.tensor(av_frame.to_rgb().to_ndarray())
                    #av2 = torch.tensor(av_frame.to_rgb(dst_colorspace='ITU709').to_ndarray())
                    #av_frames = torch.tensor(av_frame.to_rgb(dst_colorspace='ITU624').to_ndarray())
                    vision_frames = next(decoder)["data"]
                    if False:
                        if False:
                            rr = decoder._reformat(vision_frames)
                            rr = rr.reshape(av_frames.shape)
                            rr2 = _transform(rr)
                        else:
                            rr2 = vision_frames
                        print(rr2[:2, :2])
                        print(av2[:2, :2])
                        print(_transform(av_frames)[:2, :2])
                        print((_transform(av_frames) - rr2.cpu()).abs().max())
                        print((_transform(av_frames) - rr2.cpu()).abs().mean())
                        print((_transform(av_frames) - rr2.cpu()).abs().median())
                        print('----------')
                        print(torch.max(torch.abs(torch.tensor(av2).float() - rr2.cpu().float())))
                        print(torch.mean(torch.abs(torch.tensor(av2).float() - rr2.cpu().float())))
                        print(torch.median(torch.abs(torch.tensor(av2).float() - rr2.cpu().float())))
                        aa = _yuv444(av_frames).flatten(0, -2) - torch.tensor([16., 128., 128.])
                        bb = torch.tensor(av2).flatten(0, -2).float()
                        print('----------')
                        rrr = torch.linalg.lstsq(aa, bb)
                        print((bb - aa @ rrr.solution).abs().max())
                        print((bb - aa @ rrr.solution).abs().mean())
                        print((bb - aa @ rrr.solution).abs().median())

                        #print(rr[:3, :3], av_frames.shape)
                        mean_delta = torch.mean(torch.abs(av_frames.float() - rr.float()))
                        print(torch.max(torch.abs(av_frames.float() - rr.float())))
                        #mean_delta = torch.mean(torch.abs(av_frames.float() - decoder._reformat(vision_frames).float()))

                        #print((av_frames.float() - vision_frames.cpu().float()).abs().max())
                        #print((av_frames.float() - vision_frames.cpu().float()).abs().flatten().topk(10,largest=False).values)
                        #v = (av_frames.float() - vision_frames.cpu().float()).abs().flatten()
                        #v = torch.histogram(v, bins=v.unique())
                        #print(test_video, (v.hist / v.hist.sum() * 100).int())

                    av_frames_rgb = _yuv420_to_rgb(av_frames_yuv)
                    #diff = torch.abs(av_frames_rgb.floor().float() - vision_frames.cpu().float())
                    diff = torch.abs(av_frames_rgb.float() - vision_frames.cpu().float())
                    mean_delta = torch.median(diff)
                    mean_delta = torch.kthvalue(diff.flatten(), int(diff.numel() * 0.7)).values
                    if mean_delta > 16:
                        print((torch.abs(diff)).max())
                        print((torch.abs(diff)).median())
                        #v = torch.histogram(diff.flatten(), bins=diff.flatten().unique())
                        v = torch.histogram(diff.flatten(), bins=100)
                        print((v.hist / v.hist.sum() * 100).int())
                        print((v.hist / v.hist.sum() * 100).cumsum(0).int())
                        print((v.hist / v.hist.sum() * 100))
                    assert mean_delta < 16
                    #assert mean_delta < 5


if __name__ == "__main__":
    pytest.main([__file__])
