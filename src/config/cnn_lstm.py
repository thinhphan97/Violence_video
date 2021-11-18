from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "cnn_convlstm_7" # Experiment name
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 15
_C.SYSTEM.FP16 = False
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 2

_C.DIRS = CN()
_C.DIRS.DATA = "/content/"
_C.DIRS.TRAIN_DF = "/content/dataset/extract1/train/"
_C.DIRS.VALID_DF = "/content/dataset/extract1/valid/"
_C.DIRS.TEST_DF = "/content/dataset/extract1/test/"
# _C.DIRS.TRAIN_DF = "../input/abnormal-streamer/data_stream/train.pkl"
# _C.DIRS.VALID_DF = "../input/abnormal-streamer/data_stream/val.pkl"
# _C.DIRS.TEST_DF = "../input/abnormal-streamer/data_stream/test.pkl"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.MIXUP = False
_C.DATA.CM_ALPHA = 1.0
_C.DATA.IMG_SIZE = 256
_C.DATA.INP_CHANNEL = 3
_C.DATA.NUM_SLICES = 25
_C.DATA.STRIDE = 5
_C.DATA.STEP = 10

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 24

_C.INFER = CN()
_C.INFER.TTA = False

_C.OPT = CN()
# _C.OPT.OPTIMIZER = "adamw"
_C.OPT.SCHED = "cosine_warmup"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4 
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPS = 1e-3
_C.OPT.DECAY_EPOCHS = 1
_C.OPT.DECAY_RATE = 0.9


_C.LOSS = CN()
_C.LOSS.WEIGHTS = [2., 1.]

_C.MODEL = CN()
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = "se_resnext50_32x4d"

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NAME = "lstm"
_C.MODEL.DECODER.NUM_LAYERS = 1
_C.MODEL.DECODER.IN_FEATURES = 2048
_C.MODEL.DECODER.HIDDEN_SIZE = 128
_C.MODEL.DECODER.BIDIRECT = False
_C.MODEL.DECODER.DROPOUT = 0.3

_C.MODEL.CONVLSTM = CN()
_C.MODEL.CONVLSTM.INP_CHANNEL = 64
_C.MODEL.CONVLSTM.NODE_HIDDEN = 2048
_C.MODEL.CONVLSTM.NODE_HIDDEN_1 = 1024
_C.MODEL.CONVLSTM.hidden_dim = [64]
_C.MODEL.CONVLSTM.kernel_size = [(3, 3)]
_C.MODEL.CONVLSTM.num_layers = 1
_C.MODEL.CONVLSTM.batch_first = True 
_C.MODEL.CONVLSTM.bias = True 
_C.MODEL.CONVLSTM.return_all_layers = True
_C.MODEL.CONVLSTM.DROPOUT = 0.5

_C.MODEL.NUM_CLASSES = 2

_C.CONST = CN()
_C.CONST.LABELS = [
  "normal", "abnormal"
  ]