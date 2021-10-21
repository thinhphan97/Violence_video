from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "" # Experiment name
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 2

_C.DIRS = CN()
_C.DIRS.DATA = "data/rsna/"
# _C.DIRS.TRAIN_DF = "dataset/train.pkl"
# _C.DIRS.VALID_DF = "dataset/val.pkl"
# _C.DIRS.TEST_DF = "dataset/test.pkl"
_C.DIRS.TRAIN_DF = "../input/abnormal-streamer/data_stream/train.pkl"
_C.DIRS.VALID_DF = "../input/abnormal-streamer/data_stream/val.pkl"
_C.DIRS.TEST_DF = "../input/abnormal-streamer/data_stream/test.pkl"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.CUTMIX = True
_C.DATA.MIXUP = False
_C.DATA.CM_ALPHA = 1.0
_C.DATA.MEAN = []
_C.DATA.STD = []
_C.DATA.IMG_SIZE = 128
_C.DATA.INP_CHANNEL = 3
_C.DATA.NUM_SLICES = 20

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.BATCH_SIZE = 8

_C.INFER = CN()
_C.INFER.TTA = False

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.SCHED = "cosine_warmup"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPS = 1e-3
_C.OPT.DECAY_EPOCHS = 4
_C.OPT.DECAY_RATE = 1e-4


_C.LOSS = CN()
_C.LOSS.WEIGHTS = [1., 1.]

_C.MODEL = CN()
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = "se_resnext50_32x4d"

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NAME = "lstm"
_C.MODEL.DECODER.NUM_LAYERS = 2
_C.MODEL.DECODER.IN_FEATURES = 2048
_C.MODEL.DECODER.HIDDEN_SIZE = 512
_C.MODEL.DECODER.BIDIRECT = True
_C.MODEL.DECODER.DROPOUT = 0.3

# _C.MODEL.CONVLSTM.input_dim =  3
_C.MODEL.CONVLSTM = CN()
_C.MODEL.CONVLSTM.hidden_dim = [16]
_C.MODEL.CONVLSTM.kernel_size = [(3, 3)]
_C.MODEL.CONVLSTM.num_layers = 1
_C.MODEL.CONVLSTM.batch_first = True 
_C.MODEL.CONVLSTM.bias = True 
_C.MODEL.CONVLSTM.return_all_layers = False

_C.MODEL.NUM_CLASSES = 2

_C.CONST = CN()
_C.CONST.LABELS = [
  "normal", "abnormal"
  ]