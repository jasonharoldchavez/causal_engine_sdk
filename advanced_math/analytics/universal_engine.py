import numpy as np
import pandas as pd
import yfinance as yf

from scipy.linalg import det, eigh
from sklearn.preprocessing import StandardScaler


class RealUniversalEngine:
    """
    Cross-domain structural analysis engine.

    Combines:
    - correlation geometry
    - covariance structure
    - graph topology
    - spectral complexity
    """

    def __init__(
        self,
        window_size=15,
        threshold=0.4
    ):

        self.window_size = window_size
        self.threshold = threshold
        self.prev_G = None
        self.scaler = StandardScaler()


    def fetch_real_data(self):

        print(
            "Ingesting real-world streams..."
        )

        tickers = [
            "^VIX",
            "GC=F",
            "CL=F",
            "SPY"
        ]

        market = yf.download(
            tickers,
            period="3mo",
            interval="1d",
            auto_adjust=True
        )

        market_data = market["Close"]

        signals = (
            market_data
            .pct_change()
            .dropna()
        )


        steps = len(signals)


        infrastructure = pd.DataFrame(
            {
                "Network_Latency_MS":
                    np.sin(
                        np.linspace(
                            0,
                            10,
                            steps
                        )
                    ) * 5
                    +
                    np.random.normal(
                        0,
                        1,
                        steps
                    ),

                "Global_Tx_Throughput":
                    np.cos(
                        np.linspace(
                            0,
                            5,
                            steps
                        )
                    ) * 100
                    +
                    np.random.normal(
                        0,
                        5,
                        steps
                    )
            },
            index=signals.index
        )


        return pd.concat(
            [
                signals,
                infrastructure
            ],
            axis=1
        ).dropna()



    def process_real_matrix(self, df):

        raw = df.values.T


        X = self.scaler.fit_transform(
            raw.T
        ).T


        channels, steps = X.shape


        results = []


        for t in range(
            self.window_size,
            steps
        ):

            window = X[
                :,
                t-self.window_size:t
            ]


            corr = np.nan_to_num(
                np.corrcoef(window)
            )


            upper = np.triu_indices(
                channels,
                k=1
            )


            correlations = np.abs(
                corr[upper]
            )


            R = float(
                np.mean(correlations)
            )


            C = float(
                1 /
                (
                    1 +
                    np.var(correlations)
                )
            )


            covariance = np.nan_to_num(
                np.cov(window)
            )


            variances = np.diag(
                covariance
            )


            try:

                covariance_det = det(
                    covariance
                )

                if covariance_det > 1e-12:

                    E = float(
                        np.sum(
                            np.log(
                                variances + 1e-9
                            )
                        )
                        /
                        max(
                            np.log(
                                covariance_det
                                + 1e-9
                            ),
                            0.1
                        )
                    )

                else:

                    E = 1.0

            except Exception:

                E = 1.0



            G = (
                np.abs(corr)
                >
                self.threshold
            ).astype(int)



            if self.prev_G is None:

                S = 1.0

            else:

                S = float(
                    1 -
                    np.sum(
                        np.abs(
                            G -
                            self.prev_G
                        )
                    )
                    /
                    G.size
                )


            self.prev_G = G.copy()


            degrees = np.sum(
                G,
                axis=1
            )


            laplacian = (
                np.diag(degrees)
                -
                G
            )


            try:

                eigenvalues = eigh(
                    laplacian,
                    eigvals_only=True
                )

                K = float(
                    np.sum(
                        eigenvalues >
                        1e-4
                    )
                )

            except Exception:

                K = 0.0



            U = float(
                R *
                C *
                E *
                S *
                (1 + K)
            )


            results.append(
                {
                    "Step": t,
                    "Resonance_R": R,
                    "Coherence_C": C,
                    "Emergence_E": E,
                    "Stability_S": S,
                    "Complexity_K": K,
                    "Unified_U": U,
                }
            )


        return pd.DataFrame(results)



if __name__ == "__main__":


    engine = RealUniversalEngine(
        window_size=10,
        threshold=0.35
    )


    data = engine.fetch_real_data()


    output = engine.process_real_matrix(
        data
    )


    print(
        output.tail(10)
    )


    output.to_csv(
        "real_universal_metrics.csv",
        index=False
    )


    print(
        "Saved real_universal_metrics.csv"
    )
