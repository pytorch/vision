
import os.path as osp
import torchvision.io.image as image

HERE = osp.basename(osp.abspath(__file__))
TEST_ASSETS = osp.join(HERE, 'test', 'assets')
D_JPEG = osp.join(TEST_ASSETS, 'damaged_jpeg')

image.read_jpeg(osp.join(D_JPEG, 'corrupt34_2.jpg'))
