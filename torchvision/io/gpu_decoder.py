import torch


class GPUDecoder:
    def __init__(self, src_file: str, use_dev_frame: bool = True, dev: int = 0, output_format = "nv12"):
        torch.ops.load_library("build/lib.linux-x86_64-3.8/torchvision/Decoder.so")
        self.decoder = torch.classes.torchvision.GPUDecoder(src_file, use_dev_frame, dev, output_format)

    def decode_frame(self):
        return self.decoder.decode()

    def get_total_decoding_time(self):
        return self.decoder.getDecodeTime()

    def get_total_demuxing_time(self):
        return self.decoder.getDemuxTime()

    def get_demuxed_bytes(self):
        return self.decoder.getDemuxedBytes()

    def get_total_frames_decoded(self):
        return self.decoder.getTotalFramesDecoded()
