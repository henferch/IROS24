import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def softmax(x_):
    ex = np.exp(x_)
    return ex/ex.sum()

print("program start")

dataPath = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/'.format(11)

W_hr = None
with open('{}W_hr.npy'.format(dataPath), 'rb') as f:
    W_hr = np.load(f)

all_PA = []
all_ID = []



for exp in range(5,12):
    
    print("processisng experiment {}".format(exp))
    
    dataPath = '/home/hfchame/Workspace/VSCode/IROS24/src/data/data_{}/'.format(exp)
    
    W_obj = None
    with open('{}W_obj.npy'.format(dataPath), 'rb') as f:
        W_obj = np.load(f)
    
    h_human = None
    with open('{}h_human.npy'.format(dataPath), 'rb') as f:
        h_human = np.load(f)

    # fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(8, 4))
    # fig.tight_layout()

    nObj =  W_obj.shape[0]
    o_alpha = 1.0e1
    #print(W_hr.shape, W_obj.shape, h_human.shape)

    #id_i = np.array([x for x in range(4)])
    for i in range (nObj):
        x = np.matmul(W_obj,  np.matmul(W_hr, h_human[i,:]))
        h_o = softmax(o_alpha * x)
        all_PA.append(h_o)
        all_ID.append(i)


all_PA = np.vstack(all_PA)
minv = np.min(all_PA,axis=0) 
maxv = np.max(all_PA,axis=0) 
print(all_PA.shape)

all_ID = np.array(all_ID) + 1
print(all_ID.shape)        
        #u_hebbRobot += np.matmul(W_hr, h)
#df = px.data.iris()
d = {'PA1': all_PA[:,0].tolist(),\
     'PA2': all_PA[:,1].tolist(),
     'PA3': all_PA[:,2].tolist(),
     'PA4': all_PA[:,3].tolist(),
     'Landmark':all_ID.tolist()}
#print(d)
df = pd.DataFrame(data=d)


# import plotly.graph_objects as go
# fig = go.Figure(
#     data=[go.Bar(y=[2, 1, 3])],
#     layout_title_text="A Figure Displaying Itself",
#     layout = {'xaxis': {'title': 'x-label',
#                         'visible': True,
#                         'showticklabels': True},
#               'yaxis': {'title': 'y-label',
#                         'visible': False,
#                         'showticklabels': False}
#               }
# )
# fig

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['Landmark'],
                    colorscale = 'viridis',
                    showscale=False),
        dimensions = list([
            dict(range = [minv[0],maxv[0]],
                #constraintrange = [4,8],
                label = '', values = df['PA1']),
            dict(range = [minv[1],maxv[1]],
                label = '', values = df['PA2']),
            dict(range = [minv[2],maxv[2]],
                label = '', values = df['PA3']),
            dict(range = [minv[3],maxv[3]],
                label = '', values = df['PA4'])
        ])
    ),
    layout = {'xaxis': {'title': 'x-label',
                        'visible': False,
                        'showticklabels': False},
              'yaxis': {'title': 'y-label',
                        'visible': False,
                        'showticklabels': False}
              }
)

fig.update_layout(
    xaxis = {'showticklabels': False, 'tickvals': []},
    yaxis = {'showticklabels': False, 'tickvals': []},
    plot_bgcolor = 'white',
    paper_bgcolor = 'white',
    font = {'size':18, 'family':'Times New Roman'},
    margin=dict(l=70, r=50, t=50, b=50)
    #newselection = {'line':{'width':10.0}}
   # selections={'line':{'width':10.0}}
)


# fig = px.parallel_coordinates(df, color="id",range_color=[0.0,4.0], labels=['a','b','c','d'], 
#                               dimensions=['PA1', 'PA2', 'PA3', 'PA4'],
#                               color_continuous_scale=px.colors.diverging.Tropic,
#                               #color_scale = [[1,'purple'],[2,'lightseagreen'],[3,'gold'],[4,'green']],
#                               color_continuous_midpoint=4)
# #fig.update_traces(line={'width':2.5})
fig.write_image('plot_generalisation.pdf')
       
print("program end")
    

