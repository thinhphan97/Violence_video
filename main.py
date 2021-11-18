from src.config import get_cfg
from src.modeling import ConvLSTM3D
from src.data import Dataset_Custom_3d
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import numpy as np

cfg = get_cfg()
DataSet = Dataset_Custom_3d
# train_ds = DataSet(cfg, mode="train")
valid_ds = DataSet(cfg, mode="valid")
test_ds = DataSet(cfg, mode="test")
# train_loader = DataLoader(train_ds, 16, pin_memory=False, shuffle=True, drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
# valid_loader = DataLoader(valid_ds, 16, pin_memory=False, shuffle=False,drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
test_loader = DataLoader(test_ds, 16, pin_memory=False, shuffle=False,drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
model = ConvLSTM3D(cfg)
ckpt = torch.load("weights/best_convlstm3d_128_10.pth", "cpu")
model.load_state_dict(ckpt.pop('state_dict'))
model.sigmoid = nn.Softmax(dim=1)
model.eval()
model = model.cuda()

feartures = []
targets = []

for i,(image, target) in enumerate(test_loader):
    # print(image)
    image = image.cuda()
    bsize, seq_len, c, h, w = image.size()
    image = image.view(bsize * seq_len, c, h, w)
    fearture = model(image,cfg.DATA.NUM_SLICES)
    feartures.append(fearture.cpu().detach().numpy())
    targets.append(target)
    
feartures = np.concatenate(feartures,0)
print(feartures.shape)
targets = torch.cat(targets,0).numpy()
print(targets.shape)
i, _ = targets.shape
result_dict = {}
result_dict["fearture"] = feartures
result_dict["target"] = targets
# print(feartures)
# print(result_dict)
import pickle
with open('dataset/fearture_valid.pickle', 'wb') as handle:
    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('dataset/fearture_valid.pickle', 'rb') as handle:
    b = pickle.load(handle)
# print(b["fearture"][:100])
# print(b["target"][:100])    
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


skplt.metrics.plot_confusion_matrix( np.argmax(b["target"], axis = 1), np.argmax(b["fearture"],axis =1),normalize = True)
plt.savefig('plot_confusion_matrix_9_1.png')
skplt.metrics.plot_precision_recall_curve( np.argmax(b["target"], axis = 1), b["fearture"])
plt.savefig('plot_precision_recall_curve_9_1.png')
skplt.metrics.plot_roc( np.argmax(b["target"], axis = 1), b["fearture"])
plt.savefig('plot_roc_9_1.png')

target_names = ['abnormal', 'normal']
print(classification_report(np.argmax(b["target"], axis = 1), np.argmax(b["fearture"],axis =1), target_names=target_names))