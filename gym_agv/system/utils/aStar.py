# -*- coding:utf-8 _*-




import numpy as np


class Node:  # 描述AStar算法中的节点数据
    def __init__(self, name,point, g=0.0, h=0.0):
        self.name = name
        self.point = point  # 自己的坐标
        self.father = None  # 父节点
        self.g = g  # g值，g值在用到的时候会重新算
        self.h = h

class AStar:


    def __init__(self, map,site_map, startPoint, endPoint, passTag=0):

        self.openList = []
        self.closeList = []
        self.site_map=site_map
        self.map2d = {}
        self.map=map
        for key in map:
            self.map2d[key] = []
            for next in map[key]:
                self.map2d[key].append(Node(next,self.site_map[next],0,self.distance_str(next,endPoint)))
        self.startPoint = Node(startPoint,self.site_map[startPoint],0,self.distance_str(startPoint,endPoint))
        self.endPoint = Node(endPoint,self.site_map[endPoint],0,self.distance_str(endPoint,endPoint))



    def distance_str(self,a,b):
        return np.linalg.norm(self.site_map[a]-self.site_map[b],ord=2)

    def distance(self,a,b):
        return np.linalg.norm(a.point- b.point, ord=2)

    def pointInCloseList(self, point):
        if point.name in [item.name for item in self.closeList]:
                return True
        return False

    def pointInOpenList(self, point):
        if point.name in [item.name for item in self.openList]:
                return True
        return False

    def endPointInCloseList(self):
        if self.endPoint.name in [item.name for item in self.closeList]:
                return True
        return False

    def findNear(self,currentPoint):
        for next in self.map2d[currentPoint.name]:
            if next.name==self.endPoint.name:

                self.endPoint.father=currentPoint
                self.endPoint.g = currentPoint.g + self.distance(self.endPoint, currentPoint)
                self.closeList.append(self.endPoint)

                return -1

            if self.pointInCloseList(next):
                continue

            if self.pointInOpenList(next):
                if next.g>currentPoint.g:
                    next.father = currentPoint
                    next.g = currentPoint.g+self.distance(next,currentPoint)
                continue

            next.father = currentPoint
            next.g = currentPoint.g+self.distance(next,currentPoint)
            self.openList.append(next)
        return 1

    def a_start(self):
        self.openList.append(self.startPoint)
        flag = 1
        while flag:
            self.openList = sorted(self.openList,key=lambda x:x.g+x.h)
            # print([(item.name,item.g+item.h)for item in self.openList ])
            if len(self.openList)>0:
                currentPoint = self.openList[0]
                self.openList.pop(0)
                self.closeList.append(currentPoint)
            else:
                break

            flag = self.findNear(currentPoint)
            if flag==-1:
                break

        end = self.endPoint
        route = []
        route.append(end.name)
        while end!=self.startPoint:
            route.append(end.father.name)
            end = end.father
        route = [item for item in reversed(route)]
        return route

    def get_route_coords(self,route,v,rate):
        v_coord = []
        for site_index in range(len(route) - 1):
            s_site = route[site_index]
            s_coord = self.site_map[s_site]
            e_site = route[site_index + 1]
            e_coord = self.site_map[e_site]
            t = 0
            max_d = np.linalg.norm(s_coord - e_coord, ord=2)
            if e_coord[1] - s_coord[1] == 0:
                while t < max_d / v.vl:
                    v_coord.append([s_coord[0] + (e_coord[0] - s_coord[0]) / max_d * v.vl * t, s_coord[1]])
                    t += rate

            elif (e_coord[0] - s_coord[0] == 0):
                while t < max_d / v.vl:
                    v_coord.append([s_coord[0], s_coord[1] + (e_coord[1] - s_coord[1]) / max_d * v.vl * t])
                    t += rate
            else:
                while t < max_d / v.vl:
                    v_coord.append([s_coord[0] + (e_coord[0] - s_coord[0]) / max_d * v.vl * t,
                                    s_coord[1] + (e_coord[1] - s_coord[1]) / max_d * v.vl * t])
                    t += rate
        return v_coord