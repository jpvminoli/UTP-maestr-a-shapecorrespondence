# -*- coding: utf-8 -*-
"""
    Created By: JPMinoli
    Year: 2020
    This module allows to manage triangular mesh
"""
import numpy as np
import open3d as o3d
import scipy.io as sio
import spharapy.trimesh as tm
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import spharapy.spharabasis as sb
import tensorflow as tf
import trimesh

class StrMesh():
    def __init__(self, triangles=None, vertices=None, filename=None):
        """
            Initialize parameters
            Parameters:
                triangles: np array (number triangles*3)
                vertices: np array (number vertices*3)
                filename: str
            Returns: 
                Nothing
        """
        if filename is not None and triangles is None and vertices is None:
            fileExt = filename.split('.')
            l=len(fileExt)
            fileExt=fileExt[l-1]
            if fileExt=='mat':
                info=sio.whosmat(filename)[0]
                mat=sio.loadmat(filename)
                val=mat[info[0]][0,0]
                x=val['X']
                y=val['Y']
                z=val['Z']
                self.triangles=val['TRIV']-1
                self.vertices=np.concatenate((x,y,z),axis=1)
            elif fileExt=='ply':
                mesh = o3d.io.read_triangle_mesh(filename)
                self.triangles=mesh.triangles
                self.vertices=mesh.vertices
        elif filename is None and triangles is not None and vertices is not None:
            self.triangles=triangles
            self.vertices=vertices
        else:
            raise NameError('You need to set up triangles and vertices or a filename')

        self.mesh=trimesh.Trimesh(vertices=self.vertices,faces=self.triangles,process=False)
        
        self.mesho3d=o3d.geometry.TriangleMesh()
        self.mesho3d.vertices=o3d.utility.Vector3dVector(self.vertices)
        self.mesho3d.triangles=o3d.utility.Vector3iVector(self.triangles)
        self.mesho3d.compute_vertex_normals()

        # if len(self.triangles)>8000:
        #     self.mesh = self.mesh.simplify_quadric_decimation(target_number_of_triangles=8000)
        #     self.triangles=np.array(self.mesh.triangles)
        #     self.vertices=np.array(self.mesh.vertices)

        # self.spMesh = tm.TriMesh(self.triangles, self.vertices)
        # self.eigenVals=None
        # self.eigenVecs=None
        print('!Mesh load complete!')
        print('Vertices:',len(self.vertices))
        print('Triangles:',len(self.triangles))

    def getArea(self):
        return self.mesho3d.get_surface_area()
    
    def assignColor(self,colors):
        self.mesh.visual.vertex_colors=colors

    def assignHeatColor(self, minimum, maximum, value):
        colors=self.rgb(minimum, maximum, value)
        self.assignColor(colors)

    def rgb(self, minimum, maximum, value):
        value=value.reshape(-1,1)
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value-minimum) / (maximum - minimum)
        b = 255*(1 - ratio)>0
        b = (b*255*(1 - ratio)).astype(int)
        r = 255*(ratio - 1)>0
        r = (r*255*(ratio - 1)).astype(int)
        g = 255 - b - r
        return np.concatenate((r,g,b),axis=1)