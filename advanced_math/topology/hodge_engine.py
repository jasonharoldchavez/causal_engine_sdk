import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix, csc_matrix


class ASIEHodgeEngine:

    def __init__(self, graph: nx.Graph):

        self.graph = graph

        self.nodes = list(graph.nodes())
        self.edges = [tuple(sorted(e)) for e in graph.edges()]
        self.triangles = self._find_triangles()

        self.node_to_idx = {
            node: i for i, node in enumerate(self.nodes)
        }

        self.edge_to_idx = {
            edge: i for i, edge in enumerate(self.edges)
        }

    def _find_triangles(self):

        triangles = []

        for clique in nx.enumerate_all_cliques(self.graph):

            if len(clique) == 3:
                triangles.append(tuple(sorted(clique)))

            elif len(clique) > 3:
                break

        return triangles

    def compute_boundary_operators(self):

        B1 = lil_matrix(
            (len(self.nodes), len(self.edges)),
            dtype=np.float64
        )

        for idx, (u, v) in enumerate(self.edges):

            B1[self.node_to_idx[u], idx] = -1.0
            B1[self.node_to_idx[v], idx] = 1.0

        B2 = lil_matrix(
            (len(self.edges), len(self.triangles)),
            dtype=np.float64
        )

        for t_idx, (u, v, w) in enumerate(self.triangles):

            for edge, sign in [
                ((u, v), 1.0),
                ((v, w), 1.0),
                ((u, w), -1.0),
            ]:

                edge = tuple(sorted(edge))

                if edge in self.edge_to_idx:
                    B2[self.edge_to_idx[edge], t_idx] = sign

        return csc_matrix(B1), csc_matrix(B2)

    def compute_hodge_laplacian_1(self):

        B1, B2 = self.compute_boundary_operators()

        L1_down = B1.T @ B1

        if len(self.triangles):

            L1_up = B2 @ B2.T

        else:

            L1_up = csc_matrix(
                (
                    len(self.edges),
                    len(self.edges)
                ),
                dtype=np.float64
            )

        return (L1_down + L1_up).toarray()


if __name__ == "__main__":

    print("Initializing Advanced Hodge Engine...")

    G = nx.cycle_graph(6)

    G.add_edges_from([
        (0, 2),
        (2, 4),
        (4, 0)
    ])

    engine = ASIEHodgeEngine(G)

    L1 = engine.compute_hodge_laplacian_1()

    eigenvalues, _ = np.linalg.eigh(L1)

    harmonic_cycles = np.sum(
        np.isclose(
            eigenvalues,
            0.0,
            atol=1e-10
        )
    )

    print("\n--- Hodge Spectrum ---")

    print(f"Nodes      : {len(G.nodes())}")
    print(f"Edges      : {len(G.edges())}")
    print(f"Triangles  : {len(engine.triangles)}")
    print(f"L1 Shape   : {L1.shape}")
    print(f"Harmonic H1: {harmonic_cycles}")

    print("\nEigenvalues:\n")

    print(eigenvalues)
