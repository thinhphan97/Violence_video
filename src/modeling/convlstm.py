import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    # def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
    #              batch_first=False, bias=True, return_all_layers=False):
    def __init__(self,cfg):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(cfg.MODEL.CONVLSTM.kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(cfg.MODEL.CONVLSTM.kernel_size, cfg.MODEL.CONVLSTM.num_layers)
        hidden_dim = self._extend_for_multilayer(cfg.MODEL.CONVLSTM.hidden_dim, cfg.MODEL.CONVLSTM.num_layers)
        if not len(kernel_size) == len(hidden_dim) == cfg.MODEL.CONVLSTM.num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = cfg.MODEL.CONVLSTM.INP_CHANNEL
        self.hidden_dim = cfg.MODEL.CONVLSTM.hidden_dim
        self.kernel_size = cfg.MODEL.CONVLSTM.kernel_size
        self.num_layers = cfg.MODEL.CONVLSTM.num_layers
        self.batch_first = cfg.MODEL.CONVLSTM.batch_first
        self.bias = cfg.MODEL.CONVLSTM.bias
        self.return_all_layers = cfg.MODEL.CONVLSTM.return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
class ConvLSTM3D(ConvLSTM):
    def __init__(self,cfg):
        super(ConvLSTM3D, self).__init__(cfg)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.input_dim = cfg.DATA.INP_CHANNEL
        self.img_size = cfg.DATA.IMG_SIZE
        self.feature = ConvLSTM(cfg)
        self.recurrent_features = cfg.DATA.IMG_SIZE*cfg.DATA.IMG_SIZE*cfg.MODEL.CONVLSTM.hidden_dim[-1]
        self.fc = nn.Linear(in_features = 56448, out_features=cfg.MODEL.CONVLSTM.NODE_HIDDEN)
        self.fc1 = nn.Linear(cfg.MODEL.CONVLSTM.NODE_HIDDEN, self.num_classes)
        self.conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(cfg.MODEL.CONVLSTM.DROPOUT)
        nn.init.zeros_(self.fc.bias.data)
        nn.init.zeros_(self.fc1.bias.data)
        
    def forward(self, input_tensor, seq_len):
        x = input_tensor.reshape(-1, seq_len, self.input_dim, self.img_size, self.img_size )
        _,x = self.feature(x)
        x = x[0][0]
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = torch.flatten(x,1)
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.CONVLSTM = CN()
    cfg.MODEL.CONVLSTM.NODE_HIDDEN = 1024
    cfg.MODEL.CONVLSTM.input_dim =  3
    cfg.MODEL.CONVLSTM.hidden_dim = [16]
    cfg.MODEL.CONVLSTM.kernel_size = [(3, 3)]
    cfg.MODEL.CONVLSTM.num_layers = 1
    cfg.MODEL.CONVLSTM.batch_first = True 
    cfg.MODEL.CONVLSTM.bias = True 
    cfg.MODEL.CONVLSTM.return_all_layers = False
    cfg.MODEL.NUM_CLASSES = 2
    cfg.DATA = CN()
    cfg.DATA.INP_CHANNEL = 3
    cfg.DATA.IMG_SIZE = 128

    
    print(cfg)

    x = torch.rand((32, 10, 3, 128, 128))
    bsize, seq_len, c, h, w = x.size()
    x = x.view(bsize * seq_len, c, h, w)
    
    # convlstm = ConvLSTM(cfg)
    convlstm3d = ConvLSTM3D(cfg)
    last_states = convlstm3d(x, seq_len)
    h = last_states
    print(h.size())
    print(h)
    