
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#from pandas.tools.plotting import table
#print("Hello world ")

def adj_slices_sampler(nslices, l):
    """
    Arguments:
        nslices (int): number of slices sampled from a study
        l (int): length of a study
    """
    mid = np.random.randint(0, l)
    start = mid - nslices//2
#    print(start)
    end = mid + (nslices//2 + nslices%2)
#    print(end)
    idx = np.arange(start, end)
    idx += (1 - min(0, idx[0]))
    idx -= ((max(l, idx[-1]) - l)+1)
    return idx, mid


if __name__ == "__main__":
	df = pd.read_pickle("dataset/val.pkl")
	#df = df["class"]
	#df["value"] = 1
	#df = df["class"]
	#df = df.groupby(["class"]).sum()
	#print(df)
	
	#df = pd.DataFrame({'mass': [0.330, 4.87 , 5.97],
        #           'radius': [2439.7, 6051.8, 6378.1]},
        #          index=['Mercury', 'Venus', 'Earth'])
	#plot = df.plot.pie(y='mass', figsize=(5, 5))
	#plt.figure(figsize=(8,8))
	#ax1 = plt.subplot(121, aspect='equal')
	#df.plot(kind='pie', y='value',autopct='%1.1f%%', 
	#startangle=90, shadow=False, labels=['NonViolence','Violence'],
 	#legend = False, fontsize=14)
	#plt.show()
	print(df.loc[0])
