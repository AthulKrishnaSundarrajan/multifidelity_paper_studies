import matplotlib.pyplot as plt
import pickle

hf_results = 'high-fid-res.pkl'
with open(hf_results,'rb') as handle:
    hf_obj = pickle.load(handle)

lf_results = 'low-fid-res.pkl'
with open(lf_results,'rb') as handle:
    lf_obj = pickle.load(handle)


mf_results = 'multi-fid-res.pkl'
with open(mf_results,'rb') as handle:
    mf_obj = pickle.load(handle)


fig,ax = plt.subplots(1)

ax.plot(lf_obj[:,0],lf_obj[:,1],'.-',label = 'low-fid')
ax.plot(mf_obj[:,0],mf_obj[:,1],'.-',label = 'multi-fid')
ax.plot(hf_obj[:,0],hf_obj[:,1],'.-',label = 'high-fid')

ax.legend(ncol = 3)

plt.show()