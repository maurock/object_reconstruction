import numpy as np
import object_reconstruction.data.touch_charts as touch_charts
import os
import trimesh
import plotly.graph_objects as go

if __name__=='__main__':
    # Load files
    adj_info_path = os.path.join(os.path.dirname(touch_charts.__file__), 'adj_info.npy')
    adj_info = np.load(adj_info_path, allow_pickle=True).item()
    adj = adj_info['adj']

    touch_vision_path =  os.path.join(os.path.dirname(touch_charts.__file__), '101352', 'touch_vision.npy')
    touch_vision = np.load(touch_vision_path, allow_pickle=True).item()

    # Add vertices that are connected to eachother
    vertices_lines = np.array([]).reshape(0, 2, 3)
    count = 0
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] > 0 and i!=j:            
                vertices1 = touch_vision['verts'][0][i, :]
                vertices2 = touch_vision['verts'][0][j, :]
                vertices = np.vstack((vertices1, vertices2))[None, :, :]
                vertices_lines = np.vstack((vertices_lines, vertices))
                count+=1
    print(count)

    # Plot connections
    fig = go.Figure()
    for i in range(10000):
        fig.add_trace(go.Scatter3d(x=vertices_lines[i, :, 0], y=vertices_lines[i, :, 1], z=vertices_lines[i, :, 2],  mode='lines', line=dict(color='blue', width=0.5)))
    fig.add_trace(go.Scatter3d(x=touch_vision['verts'][0][:, 0], y=touch_vision['verts'][0][:, 1],z=touch_vision['verts'][0][:, 2], mode='markers', marker=dict(size=1)))
    fig.update_layout(showlegend=False)
    fig.show()