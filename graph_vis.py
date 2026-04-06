# Visualisation package
import networkx as nx

from bokeh.palettes import Spectral4
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.transform import linear_cmap


# PyG packages
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

class GraphVis:

    @staticmethod
    def default_vis(data, vis_name="default", pred=None):

        colors = []
        print("\tcolouring...", data.y.shape)
        labels = []
        lb_pred = []
        types = []

        for i in range(data.y.shape[0]):
            labels.append(data.y[i].item())
            lb_pred.append(pred[i].item())

            type = "train"
            if data.test_mask[i]:
                type = "test"
            elif data.valid_mask[i]:
                type = "valid"

            types.append(type)
            
            if pred == None:
                colors.append("blue")

                if data.test_mask[i] or data.valid_mask[i]:
                    colors[i] = "red" 
            else:
                colors.append("green")
                if pred[i] != data.y[i]:
                    colors[i] = "red"
                
                if i == 0:
                    colors[i] = "purple"

        G = to_networkx(data, to_undirected=False)

        # Show degree of each node in the graph 
        de = nx.degree(G)
        degrees = dict(de)
        nx.set_node_attributes(G, name='degree', values=degrees)

        nG = from_networkx(G, nx.spring_layout, scale=10, center=(0,0))

        nG.node_renderer.data_source.add(colors, 'color')
        nG.node_renderer.data_source.add(list(range(len(colors))), 'index')
        nG.node_renderer.data_source.add(labels, 'gt')
        nG.node_renderer.data_source.add(lb_pred, 'pred')
        nG.node_renderer.data_source.add(types, 'type')

        #Set node size and color
        nG.node_renderer.glyph = Circle(size=15, fill_color="color")

        #Choose a title!
        title = 'Visualisation of Graph-based Dataset'

        #Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Index", "@index"),
            ("Degree", "@degree"),
            ("Type", "@type"),
            ("Label", "@gt"),
            ("Pred", "@pred"),
        ]

        #Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips = HOVER_TOOLTIPS,
                    tools="pan,box_zoom, wheel_zoom,save,reset,point_draw,tap", active_scroll='wheel_zoom',
                    x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title, min_width=1600, min_height=1200)
        
        #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'

        #Set node highlight colors
        nG.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=Spectral4[1], line_width=2)
        nG.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=Spectral4[2], line_width=2)
        
        #Set edge opacity and width
        nG.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        #Set edge highlight colors
        nG.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_alpha=0.8, line_width="line_width")
        nG.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width="line_width")

        #Highlight nodes and edges
        nG.selection_policy = NodesAndLinkedEdges()
        nG.inspection_policy = NodesAndLinkedEdges()

        # Add network graph to the plot
        plot.renderers.append(nG)

        print("\tsaving bokeh graphv vis..")
        pth = f"/media/Patrik/gnn-gar/vis_{vis_name}.html"
        save(plot, pth)


    @staticmethod
    def vis(data, predicted=None, attn=None):
        is_pred = not predicted == None

        #data.x = data.x[data.train_mask]
        #data.y = data.y[data.train_mask]
        
        if is_pred:
            predicted = predicted.argmax(1)

        if not attn == None:
            weights = {}
            for i in range(data.edge_index.shape[1]):
                s = data.edge_index[0][i].item()
                t = data.edge_index[1][i].item()


                weights[(s, t)] = attn[s, t].item()
                weights[(t, s)] = attn[t, s].item()

        colors = []
        print("\tcolouring...", data.x.shape)
        for i in range(data.x.shape[0]):
            colors.append("blue")

            if data.y[i] == 1 and data.train_mask[i]:
                colors[i] = "red" 

            if is_pred and data.test_mask[i]:
                if predicted[i] == data.y[i]:
                    colors[i] = "green"
                else:
                    colors[i] = "red"   

        print("\tto_networkx...")
        gr = to_networkx(data, to_undirected=False)
        
        #nx.draw(gr, with_labels=False, font_weight='bold', node_color='orange')
        #plt.show()
        #return 


        print("\tnode degree...")
        degrees = dict(nx.degree(gr))
        lb = list(range(data.y.shape[0]))

        nx.set_node_attributes(gr, name='degree', values=degrees)
        nx.set_node_attributes(gr, name='label', values=dict(zip(lb, data.y.tolist())))
        
        if is_pred:
            nx.set_node_attributes(gr, name='pred', values=dict(zip(lb, predicted.tolist())))

        print("\tfrom_networkx...")
        network_graph = from_networkx(gr, nx.spring_layout, scale=10, center=(0,0))
        print("\t...finished networkx")

        network_graph.node_renderer.data_source.add(list(range(len(colors))), 'index')
        network_graph.node_renderer.data_source.add(colors, 'color')

        #Set node size and color
        network_graph.node_renderer.glyph = Circle(size=15, fill_color="color")

        #Choose a title!
        title = 'Visualisation of Graph-based Dataset'

        #Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Label", "@index"),
            ("Degree", "@degree"),
            ("Label", "@label"),
            ("Prediction", "@pred")
        ]

        #Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips = HOVER_TOOLTIPS,
                    tools="pan,box_zoom, wheel_zoom,save,reset,point_draw,tap", active_scroll='wheel_zoom', 
                    x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title, plot_width=1200, plot_height=800)

        #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'

        #Set node highlight colors
        network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=Spectral4[1], line_width=2)
        network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=Spectral4[2], line_width=2)
        
        #Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        #Set edge highlight colors
        network_graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_alpha=0.8, line_width="line_width")
        network_graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width="line_width")

        #Highlight nodes and edges
        network_graph.selection_policy = NodesAndLinkedEdges()
        network_graph.inspection_policy = NodesAndLinkedEdges()

        # Add edge weights based on attention matrix.
        if not attn == None:
            d = network_graph.edge_renderer.data_source.data
            d["line_width"] = [weights[edge] * 10. for edge in zip(d["start"], d["end"])]
            network_graph.edge_renderer.glyph.line_width = {'field': 'line_width'}

        #Add network graph to the plot
        plot.renderers.append(network_graph)

        print("\tsaving bokeh graphv vis..")
        save(plot)


def visualise_pred_nodes(data, name, pred):
    #for data in data_loader:
    GraphVis.default_vis(data, name, pred)
 