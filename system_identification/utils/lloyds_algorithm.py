import sys
from typing import Tuple, Sequence
from itertools import chain

import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPoint, box
from shapely.ops import polygonize
from shapely.affinity import scale
from scipy.spatial import Voronoi


class LloydsAlgorithm:
    def __init__(self, n: int, bounding_box: Tuple[float, float, float, float]):
        """
        Uses Lloyd's algorithm to approximate a centroidal Voronoi tessellation
        with n points. This results in points that are approximatly evenly distributed
        throughout the bounding box.

        References:
         [1] https://github.com/mapsense/bounded-voronoi-demo/blob/master/voronoi.py
         [2] https://en.wikipedia.org/wiki/Lloyd's_algorithm

        :param n: Number of points
        :param bounding_box: Bounding box
        """
        self.voronoi = None
        self.regions: Sequence[Polygon] = None

        self.bounding_box = Polygon([
            [bounding_box[0], bounding_box[2]],
            [bounding_box[0], bounding_box[3]],
            [bounding_box[1], bounding_box[3]],
            [bounding_box[1], bounding_box[2]],
        ])

        # We need a bounded voronoi diagram, so to get this we'll add
        # the corner points of a larger outer bounding box to the list
        # of points we're passing to the Voronoi class.
        # This should result in all regions within our bounding box having
        # no infinite vertices.
        outer_boundry: Polygon = scale(self.bounding_box, 2, 2)
        self.extra_points = [Point(point) for point in outer_boundry.exterior.coords]

        # Add n random distributed points to the list.
        points = np.vstack([
            np.random.uniform(bounding_box[0], bounding_box[1], (n,)),
            np.random.uniform(bounding_box[2], bounding_box[3], (n,))
        ]).T

        # Convert to shapely Points and build the initial voronoi
        self.points: Sequence[Point] = [Point(point) for point in points]
        self.build_voronoi()

    @property
    def points_as_array(self) -> np.ndarray:
        """
        Current points as a (n_points x 2) Numpy array.
        """
        return np.vstack([[point.x, point.y] for point in self.points])

    @property
    def points_for_voronoi(self) -> np.ndarray:
        """
        Current and extra points as a (n_points x 2) Numpy array.
        """
        return np.vstack([[point.x, point.y] for point in chain(self.points, self.extra_points)])

    @property
    def error(self):
        """
        The normalized sums of distance between the points and the current region centroids.
        """
        error = 0
        region_centroids = MultiPoint([region.centroid for region in self.regions])
        for point in self.points:
            error += point.distance(region_centroids)
        return error / len(self.points)

    def build_voronoi(self) -> None:
        """
        Build voronoi diagram.
        """
        # Build voronoi and create shapely objects from the results.
        self.voronoi = Voronoi(self.points_for_voronoi)
        lines = [
            LineString(self.voronoi.vertices[line])
            for line in self.voronoi.ridge_vertices
            if -1 not in line
        ]

        # Limit the size of the regions to the bounding box and add them to the list.
        self.regions = [self.bounding_box.intersection(region) for region in polygonize(lines)]

    def relax_points(self, iterations=15) -> None:
        """
        Relax points for the given number of iterations. Each iteration
        will move the points to a more uniformally distributed position.

        :param iterations: Number of iterations to relax the points for.
        """
        # Relax the points for the given number of iterations.
        for _ in range(iterations):
            self.points = [region.centroid for region in self.regions]
            self.build_voronoi()
