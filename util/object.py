import pandas as pd
import pathlib
import os
from torch.utils.data import Dataset
from typing_extensions import Self
from itertools import permutations
import random
from PIL import Image

class ImageData:
    def __init__(self, img_dir: pathlib.Path):
        assert img_dir.is_absolute()
        self.img_dir = img_dir
    
    def __len__(self):
        length = 0
        for f in self.img_dir.iterdir():
            if f.name.endswith('.jpg'):
                length += 1
        return length
    
    def __getitem__(self, index):
        filename = str(index) + '.jpg'

        path = self.img_dir.joinpath(filename)
        img = Image.open(path)

        return path, img

class SceneData:
    def __init__(self, scenes: pd.Series):
        self.scenes = scenes
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, index):
        scene_string = self.scenes[index]
        scene = eval(scene_string)
        return scene

class GraphData:
    def __init__(self, graphs: pd.Series):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        graph_string = self.graphs[index]
        graph = eval(graph_string)
        return graph

class Object:
    def __init__(self, target):
        if type(target[1]) is tuple:
            self.color = target[0]
            self.shape_id = target[1][0]
            self.position = target[1][1]
        else:
            self.color = target[1]
            self.shape_id = target[0]
            self.position = None
    def getX(self):
        return self.position[0] if self.position is not None else None
    def getY(self):
        return self.position[1] if self.position is not None else None
    def getZ(self):
        return self.position[2] if self.position is not None else None
    def get_shape_color(self):
        return (self.shape_id, self.color)
    
    def get_relation(self, target: Self):
        relation = [None, None, None]
        if self.getX() > target.getX():
            relation[0] = 'R'
        elif self.getX() == target.getX():
            relation[0] = 'Sx'
        else:
            relation[0] = 'L'

        if self.getY() > target.getY():
            relation[1] = 'F'
        elif self.getY() == target.getY():
            relation[1] = 'Sy'
        else:
            relation[1] = 'B'

        if self.getZ() > target.getZ():
            relation[2] = 'T'
        elif self.getZ() == target.getZ():
            relation[2] = 'Sz'
        else:
            relation[2] = 'U'

        return tuple(relation)

class Scene:
    shape_dict = {
        0: ['구', '동그라미'],
        1: ['원기둥', '동그란기둥'],
        2: ['육면체', '사각형', '상자'],
        3: ['도넛'],
        4: ['콘', '원뿔']
    }

    color_dict = {
        'red': ['빨간', '빨강색', '빨강', '붉은'],
        'blue': ['파랑', '파란색', '파란'],
        'green': ['초록', '초록색', '녹색'],
        'yellow': ['노란', '노랑색', '노랑'],
        'black': ['검정', '검은색']
    }

    def __init__(self, id, scene, img_path):
        self.id = id
        self.scene_objects = [Object(o) for o in scene]
        self.img_path = img_path
    
    def relation_to_text(self, rel):
        same_count = 0
        for i in rel:
            if 'S' in i:
                same_count += 1

        text = []

        if rel[0] == 'L':
            text.append('왼쪽')
        elif rel[0] == 'R':
            text.append('오른쪽')
        
        if rel[2] == 'T' and rel[1] == 'Sy':
            text.append('위')
        elif rel[2] == 'T':
            text.append('위쪽')
        elif rel[2] == 'U' and rel[0] == 'Sx' and rel[1] != 'Sy':
            text.append('아래쪽')
        elif rel[2] == 'U' and rel[0] == 'Sx' and rel[1] == 'Sy':
            text.append('밑')
        elif rel[2] == 'U':
            text.append('아래')

        if rel[1] == 'F':
            text.append('앞')
        elif rel[1] == 'B':
            text.append('뒤')

        text = ' '.join(text)
            
        if same_count == 2:
            text = '바로 ' + text
        if same_count == 3:
            raise Exception('something wrong!!!!!')
        
        return text
    
    def to_graph(self):
        graph = []

        node_count = (len(self.scene_objects) * (len(self.scene_objects)-1))//2

        for i in range(node_count):
            first = self.scene_objects[i]
            second = self.scene_objects[(i+1)%len(self.scene_objects)]
            node = [None, None, None]
            node[0] = first.get_shape_color()
            node[1] = second.get_shape_color()
            node[2] = first.get_relation(second)

            graph.append(tuple(node))

        return graph
    
    def generate_text(self):
        scene_graph = self.to_graph()
        sentences = [[] for _ in range(len(scene_graph))]
        for i in range(len(scene_graph)):
            item = scene_graph[i]
            first = item[0]
            second = item[1]
            relation = item[2]
            
            for first_shape in Scene.shape_dict[first[0]]:
                for first_color in Scene.color_dict[first[1]]:
                    for second_shape in Scene.shape_dict[second[0]]:
                        for second_color in Scene.color_dict[second[1]]:
                            sentence = [first_color, first_shape+'은/는', second_color, second_shape+'의', self.relation_to_text(relation)+'에', '있다.']
                            sentence = ' '.join(sentence)
                            sentences[i].append(sentence)
        return sentences