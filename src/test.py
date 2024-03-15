import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
from PIL import Image

from operator import itemgetter

a = ['a','b','c','d','e']
b = itemgetter(0,4)(a)
print(b)


# with open('data/position.npy', 'rb') as f:
#     position = np.load(f)

# with open('data/times.npy', 'rb') as f:
#     times = np.load(f)

# with open('data/humanDemo.npy', 'rb') as f:
#     humanDemo = np.load(f)

# with open('data/robotDemo.npy', 'rb') as f:
#     robotDemo = np.load(f)

# print(humanDemo)
# print(robotDemo)

ExpID = range(5,12)
ExpID = [11]

I = [340, 387, 297, 247, 269, 257, 283]
F = [1676, 1455, 1317, 1267, 1109, 1104, 1344]


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

#df = px.data.iris()
d = {'PA1': [1.0,0.9,0.95,0.85, 0.1,0.02,0.3,0.2, 1.0,0.9,0.95,0.85, 0.1,0.02,0.3,0.2],\
     'PA2': [0.1,0.02,0.3,0.2,1.0,0.9,0.95,0.85,0.1,0.02,0.3,0.2,0.01,0.025,0.35,0.02],
     'PA3': [0.01,0.025,0.35,0.02, 0.12,0.2,0.03,0.21, 1.0,0.9,0.95,0.85, 0.12,0.2,0.03,0.21],
     'PA4': [0.12,0.2,0.03,0.21, 0.1,0.02,0.3,0.2, 0.12,0.2,0.03,0.21, 1.0,0.9,0.95,0.85,],
     'id':[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]}
df = pd.DataFrame(data=d)

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['id'],
                   colorscale = 'plasma'),
        dimensions = list([
            dict(range = [0.0,1.0],
                #constraintrange = [4,8],
                label = 'Point at 1', values = df['PA1']),
            dict(range = [0.0,1.0],
                label = 'Point at 2', values = df['PA2']),
            dict(range = [0.0,1.0],
                label = 'Point at 3', values = df['PA3']),
            dict(range = [0.0,1.0],
                label = 'Point at 4', values = df['PA4'])
        ])
    )
)

fig.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white',
    font = {'size':15, 'family':'Times New Roman'},
    #newselection = {'line':{'width':10.0}}
    selections={'line':{'width':10.0}}
)


# fig = px.parallel_coordinates(df, color="id",range_color=[0.0,4.0], labels=['a','b','c','d'], 
#                               dimensions=['Point 1', 'Point 2', 'Point 3', 'Point 4'],
#                               color_continuous_scale=px.colors.diverging.Tropic,
#                               #color_scale = [[1,'purple'],[2,'lightseagreen'],[3,'gold'],[4,'green']],
#                               color_continuous_midpoint=4)
#fig.update_traces(line={'width':2.5})
fig.write_image('plot_generalisation.png')

# fig, axs = plt.subplots(1, 4, subplot_kw={'projection': '3d'}, figsize=(14, 7))
# fig.tight_layout()


# for exp in ExpID:

#     #url = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/frames'.format(exp)
#     url = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/'.format(exp)
#     files = os.listdir(url)
#     #files.sort()
#     #print(frames)
    
#     frames = []
#     for f in files :
#         if 'png' in f:
#             frames.append(f)

#     num = []
#     fDict = {}
#     for f in frames :
#         t = str(f)
#         i = int(f.split('_')[1].split('.')[0])
#         num.append(i)
#         fDict[i] = t
#     num.sort()
#     k = 0    
#     for i in num:
#         urlImg = '{}{}'.format(url, fDict[i])
#         print(urlImg)
#         #img = np.asarray(Image.open(urlImg))
#         img = cv2.imread(urlImg) 
#         axs[k].imshow(img, interpolation=None)         
#         plt.axis('off') 
#         print(urlImg)
#         k += 1
    
#     plt.show()


    #print(url, frames)