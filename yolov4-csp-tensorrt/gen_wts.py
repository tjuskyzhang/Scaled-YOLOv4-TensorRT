from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.models import *

import struct
import sys

model = Darknet('models/yolov4-csp.cfg', (512, 512))
weights = sys.argv[1]
device = torch_utils.select_device('1')
# device = torch.device('cpu')
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)

f = open('yolov4-csp.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
