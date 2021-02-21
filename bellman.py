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

    def __init__(self, filename, exchange, one, treshold):
        self.filename = filename
        self.one = one
        self.exchange = exchange
        self.df = pd.read_csv(filename).set_index("Unnamed: 0").reset_index()
        self.exchange = exchange
        self.treshold = treshold
        self.rate = 0

    def do_step(self, time, plot, calc_profit, arbitrary_volume):

        dataset = self.df.loc[(self.df['Unnamed: 0']==time)&(self.df['exchange'] == self.exchange),:]

        dataset['pair'] = dataset['pair'].str.split('-')
        dataset['pair_1'] = dataset['pair'].str[0]
        dataset['pair_2'] = dataset['pair'].str[1]
        
        self.time = time

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

        self.arbitrage(plot, calc_profit, arbitrary_volume)

    def negate_logarithm_convertor(self, graph: Tuple[Tuple[float]]) -> List[List[float]]:
        ''' log of each rate in graph and negate it'''
        result = [[-log(edge) for edge in row] for row in graph]
        return result

    def arbitrage(self, plot, calc_profit, arbitrary_volume):
        ''' Calculates arbitrage situations and prints out the details of this calculations'''

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

        print("profit to be had given volume: ", limiting_volume*rate - limiting_volume)



    def plot_graph(self, arbitrage_edges):
        G=nx.Graph()
        G.add_nodes_from(self.currencies)
        pos = nx.spring_layout(G)

        edges = []
        edge_labels = {}

        print(arbitrage_edges)

        # Now let's find the edges
        for i in range(0, (self.rates.shape[0])):
            for j in range(i, self.rates.shape[1]):
                if (self.rates[i, j] > 1e-5) & (i != j):
                    # These are the interesting edges that are actually connected
                    print("adding edge: ", self.rates[i,j])
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

        plt.savefig("simple_path.png") # save as png


if __name__ == "__main__":

    arbitrage_bot = arbitrage("./OHLCVs.csv", 'krkn', True, 1.00)
    arbitrage_bot.do_step('2020-08-17 13:30:00', True, True, 1000)
