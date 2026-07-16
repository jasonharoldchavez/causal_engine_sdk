import numpy as np
import networkx as nx


class ASIEEngine:
    """
    Adaptive Structure Intelligence Engine (ASIE)

    Base version:
    - Rolling correlation analysis
    - Correlation velocity tracking
    - Structural entropy
    - Multi-scale topology filtration
    """

    def __init__(
        self,
        num_signals,
        window_size=10,
        thresholds=None
    ):

        self.num_signals = num_signals
        self.window_size = window_size

        self.thresholds = sorted(
            thresholds if thresholds is not None
            else [0.3, 0.5, 0.7]
        )

        self.buffer = []
        self.prev_correlation = None


    def step(self, current_signals):

        current_signals = np.asarray(
            current_signals,
            dtype=float
        )

        if len(current_signals) != self.num_signals:
            raise ValueError(
                "Signal count does not match engine configuration."
            )


        self.buffer.append(
            current_signals
        )


        if len(self.buffer) < self.window_size:
            return None


        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)


        data_matrix = np.array(
            self.buffer
        )


        correlation = np.corrcoef(
            data_matrix,
            rowvar=False
        )


        correlation = np.nan_to_num(
            correlation,
            nan=0.0
        )


        # Layer 1:
        # Dynamic correlation velocity

        if self.prev_correlation is None:

            delta_rho = np.zeros_like(
                correlation
            )

        else:

            delta_rho = (
                correlation -
                self.prev_correlation
            )


        self.prev_correlation = (
            correlation.copy()
        )


        # Layer 2:
        # Structural information entropy

        entropy = (
            self._compute_structural_entropy(
                correlation
            )
        )


        # Layer 3:
        # Multi-scale topology

        topology = (
            self._apply_filtration(
                correlation
            )
        )


        return {
            "correlation_matrix": correlation,
            "delta_rho": delta_rho,
            "structural_entropy": entropy,
            "multiscale_topologies": topology
        }



    def _compute_structural_entropy(
        self,
        correlation_matrix
    ):

        weights = np.abs(
            correlation_matrix.copy()
        )

        np.fill_diagonal(
            weights,
            0
        )


        total = np.sum(weights)


        if total <= 0:
            return 0.0


        probabilities = (
            weights /
            total
        )


        mask = probabilities > 0


        entropy = -np.sum(
            probabilities[mask]
            *
            np.log(
                probabilities[mask]
            )
        )


        return float(entropy)



    def _apply_filtration(
        self,
        correlation_matrix
    ):

        weights = np.abs(
            correlation_matrix.copy()
        )

        np.fill_diagonal(
            weights,
            0
        )


        layers = {}


        possible_edges = (
            self.num_signals *
            (self.num_signals - 1)
        )


        for idx, theta in enumerate(
            self.thresholds
        ):

            adjacency = (
                weights >= theta
            ).astype(int)


            G = nx.from_numpy_array(
                adjacency
            )


            try:
                loops = len(
                    nx.cycle_basis(G)
                )

            except Exception:
                loops = 0


            layers[
                f"scale_{idx+1}_theta_{theta}"
            ] = {

                "components":
                    nx.number_connected_components(
                        G
                    ),

                "loops":
                    loops,

                "edge_density":
                    float(
                        np.sum(adjacency) /
                        possible_edges
                    )
                    if possible_edges > 0
                    else 0.0
            }


        return layers



if __name__ == "__main__":

    np.random.seed(42)


    engine = ASIEEngine(
        num_signals=5,
        window_size=10
    )


    data = np.random.randn(
        15,
        5
    )


    print(
        "Running ASIE verification..."
    )


    for t, frame in enumerate(data):

        result = engine.step(frame)


        if result:

            print(
                "\nStep:",
                t
            )

            print(
                "Entropy:",
                result["structural_entropy"]
            )

            print(
                "Max velocity:",
                np.max(
                    np.abs(
                        result["delta_rho"]
                    )
                )
            )
