To import the rural street networks from the provided pickle files, use the following code

import pickle as pkl
G = pkl.load(open("fname", "rb"))

where 'fname' is the filename of the pickle file.

The networkX graph G then contains information about the nodes, indexed by the OSM ID and with attribute 'pos' denoting the latitude and longitude of the node, and about the edges, including the length of the edges as an attribute 'weight'. 
Keep in in mind that one degree longitude describes a different absolute length depending on the latitude. Otherwise, you do not need a specific projection to plot the network since the region is sufficiently small and distortions are negligible.

The networks cover
(i) the region around Rethem und Hülsen in Lower Saxony.
(ii) the villages Roßwein and Nossen in Saxony.
(iii) the vilages Epfenbach, Spechbach, Lobenfeld, Waldwimmersbach, Michelbach, Reichartshausen in Baden-Württemberg.

You can check the exact positions using the longitude/latitude attributes of the nodes.




To download your own rural networks, follow these steps:

1. Create the region you want to download as a geojson polygon using geojson.io
2. Download the street network using OSMnx in Python using the code provided in 'street_network_from_json.py'.

The networkX graph G then contains information about the nodes, indexed by the OSM ID and with attributes 'x' and 'y' denoting the latitude and longitute of the node, and about the edges, including the length of the edges as an attribute 'length'. 

You can access these attributes via 
    G.[j][i]['length']
for the attribute 'length' of the link from 'j' to 'i' and
    G.nodes()[j]['x']
for the attribute 'x' of node j.
