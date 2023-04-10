# -*- coding:utf-8 _*-

import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpathes
import matplotlib.animation as animation
global coords2
global point_ani
global max_len
global point_ani1
global point_ani2
global point_ani3
global text_pt1
global text_pt2
global text_pt3
global text_pt
class DynamciRender():
    def __init__(self,map,vehicles):
        self.map = map
        self.vehicles = vehicles

    def get_positions(self):
        positions = {}
        for i,v in enumerate(self.vehicles):
            positions[i]=[]
            for site_index in range(len(v.running_route)-1):
                if site_index+1==len(v.running_route)-1:
                    positions[i] += self.map.running_map[v.running_route[site_index]][v.running_route[site_index + 1]]["route"]
                else:
                    positions[i]+=self.map.running_map[v.running_route[site_index]][v.running_route[site_index+1]]["route"][:-1]
        coords = self.cal_position(positions)
        global max_len
        max_len = 0
        for v_c in coords:
            max_len=max(max_len,len(v_c))
            print(len(v_c))
        for v_c in coords:
            while len(v_c)<max_len:
                v_c.append(v_c[-1])
        coords = np.array(coords)
        test = coords.swapaxes(0, 1)
        global coords2
        coords2 = coords.swapaxes(0,1)

    def cal_position(self,positions):
        coords = []
        for i, v in enumerate(self.vehicles):
            v_coord = []
            for site_index in range(len(positions[i])-1):
                s_site = positions[i][site_index]
                s_coord = self.map.node_map[s_site]
                e_site = positions[i][site_index+1]
                e_coord = self.map.node_map[e_site]
                t = 0
                max_d = np.linalg.norm(s_coord - e_coord, ord=2)
                if e_coord[1]-s_coord[1]==0:
                    while t<max_d/v.vl:
                        v_coord.append([s_coord[0]+(e_coord[0]-s_coord[0])/max_d*v.vl*t,s_coord[1]])
                        t+=1

                elif (e_coord[0]-s_coord[0]==0):
                    while t<max_d/v.vl:
                        v_coord.append([s_coord[0],s_coord[1]+(e_coord[1]-s_coord[1])/max_d*v.vl*t])
                        t+=1
                else:
                    while t<max_d/v.vl:
                        v_coord.append([s_coord[0]+(e_coord[0]-s_coord[0])/max_d*v.vl*t,s_coord[1]+(e_coord[1]-s_coord[1])/max_d*v.vl*t])
                        t+=1

            coords.append(v_coord)
        return coords

    def render(self):
        self.get_positions()
        global coords2
        x = coords2[:,:,0]
        y = coords2[:,:,1]

        _,ax = plt.subplots(tight_layout=True)
        all_node = np.array([np.array(item.get_coord()) for item in self.map.nodes if not re.search(r"p\d",item.name)])
        rect_all_node = np.array([np.array(item.get_coord()) for item in self.map.nodes if not re.search(r"p\d", item.name) and item.name!="carport" and item.name!="warehouse"])
        plt.scatter(all_node[:, 0], all_node[:, 1],c="b",alpha=0.5)
        l=10
        w=7

        for i in range(len(rect_all_node)):
            height_y=4
            xy=(rect_all_node[i, 0]-l/2, rect_all_node[i, 1]+height_y)
            rect = mpathes.Rectangle(xy, l, w, color='b',fill=False)
            ax.add_patch(rect)
            coords = np.array([[rect_all_node[i, 0],rect_all_node[i, 1]],[rect_all_node[i, 0],rect_all_node[i, 1]+height_y]])

            plt.plot(coords[:,0],coords[:,1],c="b",alpha=0.5)
        for key in self.map.topology:
            for next in self.map.topology[key]:
                line = []
                line.append(self.map.node_map[key])
                line.append(self.map.node_map[next])
                line=np.array(line)
                plt.plot(line[:, 0], line[:, 1], c="b", linewidth=1)
        for node in self.map.node_map:
            # print(re.search(r"p\d",node))
            if re.search(r"p\d",node):
                continue
            elif node=="carport" :
                ll=20
                lw=6
                height_y = lw
                plt.text(self.map.node_map[node][0] - ll/ 2, self.map.node_map[node][1] + lw+1, node, fontsize=12)
                xy = (self.map.node_map[node][0]- ll/ 2, self.map.node_map[node][1] + height_y)
                rect = mpathes.Rectangle(xy, ll, lw, color='b', fill=False)
                ax.add_patch(rect)
                coords = np.array([[self.map.node_map[node][0], self.map.node_map[node][1]], [self.map.node_map[node][0], self.node_map[node][1] + height_y]])
                plt.plot(coords[:, 0], coords[:, 1], c="b",alpha=0.5)
            elif  node=="warehouse":
                ll=27
                lw=6
                height_y = lw
                shift_x = ll/ 2-6
                plt.text(self.map.node_map[node][0] - shift_x, self.map.node_map[node][1] + lw+1, node, fontsize=12)
                xy = (self.map.node_map[node][0]- shift_x, self.map.node_map[node][1] + lw)
                rect = mpathes.Rectangle(xy, ll+1, lw, color='b', fill=False)
                ax.add_patch(rect)
                coords = np.array([[self.map.node_map[node][0], self.map.node_map[node][1]], [self.map.node_map[node][0], self.node_map[node][1] + height_y]])
                plt.plot(coords[:, 0], coords[:, 1], c="b",alpha=0.5)
            else:
                plt.text(self.map.node_map[node][0]-l/2+1, self.map.node_map[node][1]+5, node, fontsize=13)

        global point_ani1
        global point_ani2
        global point_ani3
        global text_pt1
        global text_pt2
        global text_pt3
        global text_pt
        text_pt = plt.text(4, 0.8, '', fontsize=16)
        point_ani1, = plt.plot(x[0], y[0], "o", markersize=20)
        text_pt1 = plt.text(x[0][0], y[0][0], 'v0', fontsize=16)
        point_ani2, = plt.plot(x[0], y[0], "o", markersize=20)
        text_pt2 = plt.text(x[0][0], y[0][0], 'v1', fontsize=16)
        point_ani3, = plt.plot(x[0], y[0], "o", markersize=20)
        text_pt3 = plt.text(x[0][0], y[0][0], 'v2', fontsize=16)

        global max_len
        ani = animation.FuncAnimation(fig, self.update, np.arange(0, max_len), interval=1, blit=True)
        # ani.save('nvf.gif', writer='imagemagick', fps=50)
        plt.show()

    def update(self,num):
        global coords2
        global point_ani1
        global point_ani2
        global point_ani3
        global text_pt1
        global text_pt2
        global text_pt3
        point_ani1.set_data(coords2[:,:,0][num][0], coords2[:,:,1][num][0])
        point_ani2.set_data(coords2[:, :, 0][num][1], coords2[:, :, 1][num][1])
        point_ani3.set_data(coords2[:, :, 0][num][2], coords2[:, :, 1][num][2])
        # print(coords2[:,:,0][num][0], coords2[:,:,1][num][0])
        text_pt1.set_position((coords2[:,:,0][num][0], coords2[:,:,1][num][0]))
        text_pt2.set_position((coords2[:, :, 0][num][1], coords2[:, :, 1][num][1]))
        text_pt3.set_position((coords2[:, :, 0][num][2], coords2[:, :, 1][num][2]))
        return point_ani1,point_ani2,point_ani3,text_pt1,text_pt2,text_pt3,