import networkx as nx
from matplotlib import pyplot as plt


def visualize(routing, pos=None):
    G = nx.DiGraph()

    # Add edges for each route
    for route in routing:
        for i in range(len(route) - 1):
            G.add_edge(route[i], route[i + 1])

    # You might want to specify positions for each node if you have them.
    # For a simple automatic layout, you can use:
    if pos is None:
        pos = nx.spring_layout(G, k=1)  # Positions for all nodes

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Prepare a list of colors for the routes
    # colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

    # Draw edges for each route in a different color
    for idx, route in enumerate(routing):
        # Select color for this route
        # color = colors[idx % len(colors)]  # Cycle through colors if not enough
        # Draw edges for this route
        edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=f"C{idx}", width=2, arrows=True)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Show the plot
    plt.axis('off')  # Turn off the axis
    return pos
