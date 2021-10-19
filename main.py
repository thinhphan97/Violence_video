
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from src.data import Dataset_Custom_3d

if __name__ == "__main__":
	# df = pd.read_pickle("dataset/val.pkl")
	# #df = df["class"]
	# #df["value"] = 1
	# #df = df["class"]
	# #df = df.groupby(["class"]).sum()
	# #print(df)
	
	# #df = pd.DataFrame({'mass': [0.330, 4.87 , 5.97],
    #     #           'radius': [2439.7, 6051.8, 6378.1]},
    #     #          index=['Mercury', 'Venus', 'Earth'])
	# #plot = df.plot.pie(y='mass', figsize=(5, 5))
	# #plt.figure(figsize=(8,8))
	# #ax1 = plt.subplot(121, aspect='equal')
	# #df.plot(kind='pie', y='value',autopct='%1.1f%%', 
	# #startangle=90, shadow=False, labels=['NonViolence','Violence'],
 	# #legend = False, fontsize=14)
	# #plt.show()
	# print(df.loc[0])
	from src.config import get_cfg
	from src.modeling import ConvLSTM3D
	import torch
	cfg = get_cfg()

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
