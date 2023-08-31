from abc import ABC, abstractmethod
from typing import Type, Tuple
from multiprocessing import Pool, cpu_count

from numba import njit
import numpy as np
import numpy.linalg as linalg


Vector3 = Type[Tuple[float, float, float]]

# def norm(v: Vector3):
#     return ((v[0] ** 2) + (v[1] ** 2) + (v[2] ** 2)) ** (1/2)

# primitives


class Shape(ABC):
    
    @abstractmethod
    def sdf(self, p: Vector3):
        """
        The signed distance field function of the shape volume
        """
        pass
    
    # parallelized code
    def bake_sdf_for_slice(self, args):
        xi, yi, zsize, resolution, bound = args
        slice_data = np.zeros(zsize)
        
        for zi in range(zsize):
            x = float(bound[0] + xi / resolution)
            y = float(bound[2] + yi / resolution)
            z = float(bound[4] + zi / resolution)
            
            slice_data[zi] = self.sdf(np.array([x, y, z]))
        
        return slice_data


    def parallel_bake_sdf(self, resolution: int, bound: tuple):
        """
        Parallelized version of bake_sdf. Generally performs faster.

        Due to parallelization overhead, this method may be slower than bake_sdf for low resolutions
        """

        xsize = (bound[1] - bound[0]) * resolution
        ysize = (bound[3] - bound[2]) * resolution
        zsize = (bound[5] - bound[4]) * resolution

        num_processes = cpu_count()  # Number of available CPU cores
        pool = Pool(processes=num_processes)

        def slice_params():
            for yi in range(ysize):
                for xi in range(xsize):
                    yield (xi, yi, zsize, resolution, bound)

        sdf_slices = pool.map(self.bake_sdf_for_slice, slice_params())
        pool.close()
        pool.join()

        arr = np.stack(np.split(np.array(sdf_slices), ysize))

        return arr


    def bake_sdf(self, resolution: int, bound: tuple):
        """
        resolution: the number of pixel units per meter
        bound: the bounding box in the format (xmin, xmax, ymin, ymax, zmin, zmax), where mins are inclusive and maxes are exclusive, in units of meter
        """

        xsize = (bound[1] - bound[0]) * resolution
        ysize = (bound[3] - bound[2]) * resolution
        zsize = (bound[5] - bound[4]) * resolution

        arr = np.zeros((xsize, ysize, zsize))

        for xi in range(xsize):
            for yi in range(ysize):
                for zi in range(zsize):
                    x = float(bound[0] + xi / resolution)
                    y = float(bound[2] + yi / resolution)
                    z = float(bound[4] + zi / resolution)

                    arr[xi, yi, zi] = self.sdf(np.array([x, y, z]))

        return arr


class Sphere(Shape):

    def __init__(self, radius: float):
        self.radius = radius
    

    def sdf(self, p: Tuple[float, float, float]):
        return Sphere._sdf(self.radius, p)
    

    @njit
    def _sdf(radius, p: Tuple[float, float, float]):
        return linalg.norm(p) - radius


class Box(Shape):

    def __init__(self, size: Vector3):
        self.size = size
    

    def sdf(self, p: Tuple[float, float, float]):
        return Box._sdf(self.size, p)
    

    @njit
    def _sdf(size, p: Tuple[float, float, float]):
        q = np.absolute(p) - size
        return linalg.norm(np.fmax(q, np.zeros(3))) + min(max(q[0], q[1], q[2]), 0)
    

class Union(Shape):

    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b
    

    def sdf(self, p: Tuple[float, float, float]):
        return min(self.a.sdf(p), self.b.sdf(p))