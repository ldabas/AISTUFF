from typing import Tuple, List
from math import log
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

class arbitrage:
    """
    Class to calculate arbitrage opportunities given OHLVs dataset. Also has the capability
    of using limit order book dataset or arbitrary volume to calculate profits based on 
    available volume.

    There is a lot to be optimized here, and it is terribly inefficient. This is a proof of concept
    rather than a finalized product.

    Data files should be available in current folder to run!

    Bellman-Ford algorithm inspiration:
    - https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/
    - https://www.programiz.com/dsa/bellman-ford-algorithm
    - https://reasonabledeviations.com/2019/03/02/currency-arbitrage-graphs/
    - https://medium.com/@anilpai/currency-arbitrage-using-bellman-ford-algorithm-8938dcea56ea
    - https://gist.github.com/anilpai/fe4e11b5c59d8c02813900813396400b
    - https://reasonabledeviations.com/2019/04/21/currency-arbitrage-graphs-2/
    - https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
    """

    def __init__(self, filename, exchange, one, treshold):
        """
        Constructor

        Args:
            filename (string): file name of OHLCVs data
            exchange (string): name of exchange to calculate arbitrage opportunities for
            one (bool): if set to True, calculates for one exchange, False not yet implemented
            treshold (float): only show profit rate when it is above this treshold
        """
        self.filename = filename
        self.one = one
        self.exchange = exchange
        self.df = pd.read_csv(filename).set_index("Unnamed: 0").reset_index()
        self.exchange = exchange
        self.treshold = treshold
        self.rate = 0

    def do_step(self, time, plot, calc_profit, arbitrary_volume):
        """
        Does the Bellman-Ford algorithm for a single timestep. Generates graph and calculates shortest path
        based on that graph. Prints out these shortest paths and profit rates following these paths.

        Args:
            time (string): time to evaluate algorithm for in string format
            plot (bool): Whether to plot graph
            calc_profit (bool): whether to calculate profits as well given volume
            arbitrary_volume (number): if set to 0 uses LOB to calculate profit given volume, otherwise uses this value as 
                                       volume
        """ 
        
        # Select this timestep in the dataset
        dataset = self.df.loc[(self.df['Unnamed: 0']==time)&(self.df['exchange'] == self.exchange),:]

        # Split up the pairs into distinctive columns
        dataset['pair'] = dataset['pair'].str.split('-')
        dataset['pair_1'] = dataset['pair'].str[0]
        dataset['pair_2'] = dataset['pair'].str[1]
        
        self.time = time

        # Take only necessary columns from dataframe, cleaner to do this in initial setup (__init__), however no time
        if self.one == True:
            graph_df = dataset.loc[:,['price','pair_1','pair_2']]
        else:
            graph_df = dataset.loc[:,['price','exchange', 'pair_1','pair_2']]
            graph_df['pair_1'] = graph_df['pair_1'] + '_'  + graph_df['exchange']
            graph_df['pair_2'] = graph_df['pair_2'] + '_'  + graph_df['exchange']

        pair1_unique = graph_df['pair_1'].unique() 
        pair2_unique = graph_df['pair_2'].unique() 
        labels = np.unique(np.concatenate((pair1_unique,pair2_unique),axis = 0))
        self.currencies = labels
        self.graph = pd.DataFrame(index = labels, columns = labels)

        # Make sure we can go both ways, e.g. btc-eth -> 1/eth-btc.
        for row in labels:
            for col in labels:
                if graph_df.loc[(graph_df['pair_1'] == row) & (graph_df['pair_2'] == col)]['price'].values.size > 0:
                    self.graph.loc[row,col] = graph_df.loc[(graph_df['pair_1'] == row) & (graph_df['pair_2'] == col)]['price'].values[0]
                elif graph_df.loc[(graph_df['pair_1'] == col) & (graph_df['pair_2'] == row)]['price'].values.size > 0:
                    self.graph.loc[row,col] = 1/(graph_df.loc[(graph_df['pair_1'] == col) & (graph_df['pair_2'] == row)]['price'].values[0])
                
                r = row.split('_')[0]
                c = col.split('_')[0]
                if self.one == False:
                    if r == c:
                        self.graph.loc[row,col] = 1.0001

        np.fill_diagonal(self.graph.values, 1)
        self.graph = self.graph.replace(np.nan, 1e-6)
        self.rates = self.graph.values

        # Do the actual arbitrage
        self.arbitrage(plot, calc_profit, arbitrary_volume)

    def negate_logarithm_convertor(self, graph: Tuple[Tuple[float]]) -> List[List[float]]:
        """
        Calculates the logarithm of all values in the graph and negates them. Bellman ford algortihm can better
        handle the values then.

        Args:
            graph (Tuple[Tuple[float]]): Input graph

        Returns:
            List[List[float]]: Log-negated graph
        """
        result = [[-log(edge) for edge in row] for row in graph]
        return result

    def arbitrage(self, plot, calc_profit, arbitrary_volume):
        """
        Calculates the arbitrage opportunities given

        Args:
            plot (bool): Whether or not to plot resulting graph
            calc_profit (bool): Whether or not to calculate profits given some volume
            arbitrary_volume (bool): if set to 0 uses LOB to calculate profit given volume, otherwise uses this value as 
                                     volume
        """

        trans_graph = self.negate_logarithm_convertor(self.rates)

        # Pick any source vertex -- we can run Bellman-Ford from any vertex and get the right result

        source = 0
        n = len(trans_graph)
        min_dist = [float('inf')] * n

        pre = [-1] * n
        
        min_dist[source] = source
        arbitrage_edges = set()

        # 'Relax edges |V-1| times'
        for _ in range(n-1):
            for source_curr in range(n):
                for dest_curr in range(n):
                    if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                        min_dist[dest_curr] = min_dist[source_curr] + trans_graph[source_curr][dest_curr]
                        pre[dest_curr] = source_curr

        # if we can still relax edges, then we have a negative cycle
        for source_curr in range(n):
            for dest_curr in range(n):
                if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                    # negative cycle exists, and use the predecessor chain to print the cycle
                    print_cycle = [dest_curr, source_curr]
                    # Start from the source and go backwards until you see the source vertex again or any vertex that already exists in print_cycle array
                    while pre[source_curr] not in  print_cycle:
                        print_cycle.append(pre[source_curr])
                        source_curr = pre[source_curr]
                    print_cycle.append(pre[source_curr])

                    arbitrage_edges_current_cycle = set()

                    temp = 1
                    print_cycle = np.flip(print_cycle)
                    for i in range(len(print_cycle)-1):
                        temp *= self.graph.loc[self.currencies[print_cycle[i]],self.currencies[print_cycle[i+1]]]

                    if temp > self.treshold:
                        print("Arbitrage Opportunity: \n")
                        print(" --> ".join([self.currencies[p] for p in print_cycle[::-1]]))
                        print("Return for Opportunity:\t", temp)

                    if plot and (temp > self.treshold):
                        for i in range(len(print_cycle)-1):
                            arbitrage_edges.add((self.currencies[print_cycle[i]],self.currencies[print_cycle[i+1]]))

                            arbitrage_edges_current_cycle.add((self.currencies[print_cycle[i]],self.currencies[print_cycle[i+1]]))

                        if calc_profit:
                            self.calculate_profit('LOBs.csv', arbitrage_edges_current_cycle, temp, arbitrary_volume)

        self.arbitrage_edges = arbitrage_edges

        if plot:
            self.plot_graph(arbitrage_edges)

    def calculate_profit(self, orderbook_filename, arbitrage_edges_current_cycle, rate, arbitrary_volume = 0):
        """
        Function to calculate maximum possible profit given an order book file, one arbitrage triangle's edges, and rate.
        Essentially calculates how much volume could be traded based on the order book and uses that to calculate total profit
        to be had.

        Another mode of execution simply requires the user to insert an arbitrary or external volume and profits will
        be calculated as if that complete volume can be traded at the given price.

        Prints out profit.

        Args:
            orderbook_filename (string): file name of the limit order book
            arbitrage_edges_current_cycle (set of tuples): set with all the currency tuples in an arbitrage cycle
            rate (float): profit rate for this cycle
            arbitrary_volume (int, optional): If set to value other than 0 will use that volume as volume traded in this
                                              cycle. Defaults to 0.
        """
        if arbitrary_volume == 0:
            lob = pd.read_csv(orderbook_filename).set_index("Unnamed: 0").reset_index()
            lob_time = lob.loc[(lob['Unnamed: 0']==self.time)&(lob['exchange'] == self.exchange),:]

            lob_time['pair'] = lob_time['pair'].str.split('-')
            lob_time['pair_1'] = lob_time['pair'].str[0]
            lob_time['pair_2'] = lob_time['pair'].str[1]

            limiting_volume = 1e10
            for arbitrage_edge in arbitrage_edges_current_cycle:
                vol_for_price = lob_time.loc[(lob_time['pair_1'] == arbitrage_edge[0])][(lob_time['pair_2'] == arbitrage_edge[1])]['ask_volume0_1']
                if vol_for_price.empty:
                    vol_for_price = lob_time.loc[(lob_time['pair_2'] == arbitrage_edge[0])][(lob_time['pair_1'] == arbitrage_edge[1])]['ask_volume0_1']


                if (not vol_for_price.empty):
                    if (vol_for_price.values[0] < limiting_volume):
                        limiting_volume = vol_for_price.values[0]

            if limiting_volume == 1e10:
                limiting_volume = 0

        if arbitrary_volume > 0:
            limiting_volume = arbitrary_volume

        print("profit to be made given volume in displayed currency: ", round(limiting_volume*rate - limiting_volume, 5))



    def plot_graph(self, arbitrage_edges):
        """
        Plots graph with nodes as currencies, edges as currency pairs, and pair prices as edge weights

        Args:
            arbitrage_edges (set of tuples): set with all the currency tuples in an arbitrage cycle
        """
        G=nx.Graph()
        G.add_nodes_from(self.currencies)
        pos = nx.spring_layout(G)

        edges = []
        edge_labels = {}

        # Now let's find the edges
        for i in range(0, (self.rates.shape[0])):
            for j in range(i, self.rates.shape[1]):
                if (self.rates[i, j] > 1e-5) & (i != j):
                    # These are the interesting edges that are actually connected
                    if (self.currencies[i], self.currencies[j]) in arbitrage_edges or \
                        (self.currencies[j], self.currencies[i]) in arbitrage_edges:
                        G.add_edge(self.currencies[i], self.currencies[j], color='r')
                    else:
                        G.add_edge(self.currencies[i], self.currencies[j], color='b')
                    edge_labels[(self.currencies[i], self.currencies[j])] = round(self.rates[i,j],3)

        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]

        nx.draw(G, pos, edge_color=colors, width=1, linewidths=1, node_size=500, node_color='pink', \
                alpha=0.9, labels={node:node for node in G.nodes()})


        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.savefig("output_graph.png") # save as png


if __name__ == "__main__":

    # Edit with for loop over times to get profits over time period. Currently as demo implemented just one timestep
    # to not clutter stdout.
    arbitrage_bot = arbitrage("./OHLCVs.csv", 'krkn', True, 1.002)
    arbitrage_bot.do_step('2020-08-17 13:30:00', True, True, 500)
