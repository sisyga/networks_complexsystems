import osmnx as ox
from shapely.geometry import Polygon
import json

def get_polygon_from_json(path_to_json):
    """
    Reads json at path. json can be created at http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    coordinates = data['features'][0]['geometry']['coordinates'][0]
    coordinates = [(item[0], item[1]) for item in coordinates]
    polygon = Polygon(coordinates)
    return polygon


def get_polygons_from_json(path_to_json):
    """
    Reads json at path. json can be created at http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    polygons = []
    for d in data['features']:
        coordinates = d['geometry']['coordinates'][0]
        coordinates = [(item[0], item[1]) for item in coordinates]
        polygons.append(Polygon(coordinates))
    return polygons


def main():
    path_to_json = '...'
    # Wenn es nur ein Polygon ist.
    polygon = get_polygon_from_json(path_to_json)
    # Wenn es mehr als ein Polygon ist.
    # polygons = get_polygons_from_json(path_to_json)

    G = ox.graph_from_polygon(polygon, network_type='drive')
