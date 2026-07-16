import numpy as np
import gudhi as gd


class ASIETopologicalEngine:
    """
    Advanced Persistent Homology Engine
    Computes Vietoris-Rips filtrations and persistence diagrams.
    """

    def __init__(self, max_edge_length=2.0, max_dimension=2):
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension

    def compute_persistence(self, point_cloud):
        """
        Build a Vietoris-Rips complex and compute persistence.
        """
        rips_complex = gd.RipsComplex(
            points=point_cloud,
            max_edge_length=self.max_edge_length
        )

        simplex_tree = rips_complex.create_simplex_tree(
            max_dimension=self.max_dimension
        )

        persistence = simplex_tree.persistence()

        return simplex_tree, persistence

    def extract_structural_features(self, simplex_tree):
        """
        Attempt to extract Betti numbers from the simplex tree.
        Falls back gracefully if the installed Gudhi version
        exposes a different API.
        """
        try:
            return simplex_tree.betti_numbers()
        except AttributeError:
            return []

    def summary(self, simplex_tree, persistence):
        return {
            "simplices": simplex_tree.num_simplices(),
            "vertices": simplex_tree.num_vertices(),
            "features": len(persistence)
        }


if __name__ == "__main__":

    print("Initializing ASIETopologicalEngine...")

    n_points = 300

    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)

    R = 2.0
    r = 0.5

    x = (R + r*np.cos(phi))*np.cos(theta)
    y = (R + r*np.cos(phi))*np.sin(theta)
    z = r*np.sin(phi)

    point_cloud = np.vstack((x, y, z)).T

    engine = ASIETopologicalEngine(
        max_edge_length=1.5,
        max_dimension=2
    )

    tree, persistence = engine.compute_persistence(point_cloud)

    print("\n--- Topological Inspection ---")
    print(f"Simplices : {tree.num_simplices()}")
    print(f"Vertices  : {tree.num_vertices()}")

    print("\nFirst Persistence Intervals:\n")

    for dim, interval in persistence[:10]:
        print(
            f"H{dim}: "
            f"Birth={interval[0]:.4f}, "
            f"Death={interval[1]:.4f}"
        )

    print("\nBetti Numbers:")

    print(engine.extract_structural_features(tree))
