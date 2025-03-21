import logging

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

from explanation.explainers.data_explainers.base_data_explainer import BaseDataExplainer


class Macrobase(BaseDataExplainer):
    """MacroBase class.

    See https://cs.stanford.edu/~deepakn/assets/papers/macrobase-sigmod17.pdf for more details.

    Args:
        n_bins: number of bins to use for histogram-based discretization.
        min_support: outlier support threshold.
        min_risk_ratio: relative risk ratio threshold.
    """

    fitting_step = False

    def __init__(
        self, n_bins: int, min_support: int, min_risk_ratio: float, **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.n_bins = n_bins
        self.min_support = min_support
        self.min_risk_ratio = min_risk_ratio

    def predict_instance(
        self, normal_records: np.array, anomalous_records: np.array
    ) -> dict:
        """MacroBase implementation.

        An "explanation" is defined as a set of relevant features and corresponding anomalous
        value intervals (included start, excluded end). A single anomalous interval is provided
        for each feature.

        Internally, these correspond to the most relevant itemset (i.e., set of items),
        each subset of which has sufficient outlier support and relative risk ratio.

        Features of the provided records are assumed continuous, and therefore require an
        additional "discretization" step (here performed using histogram-based binning on the
        normal records). In this context, what defines an item is the association of a feature
        and a value range.
        """
        # get boolean transactions with items of sufficient outlier support and relative risk ratio
        (
            normal_transactions,
            anomalous_transactions,
            features_bins,
        ) = self.get_boolean_transactions(normal_records, anomalous_records)
        # get itemsets with sufficient outlier support using FP-Growth
        itemsets_df = fpgrowth(
            anomalous_transactions, min_support=self.min_support, use_colnames=True
        )
        if itemsets_df.empty:
            logging.warning("No explanation found for the sample.")
            return {"feature_to_importance": dict(), "feature_to_intervals": dict()}

        # add relative risk ratio and cardinality information to the itemsets
        risk_ratios = []
        n_items = []
        for itemset in itemsets_df["itemsets"]:
            risk_ratios.append(
                self.get_itemset_risk_ratio(
                    itemset, normal_transactions, anomalous_transactions
                )
            )
            n_items.append(len(itemset))
        itemsets_df = itemsets_df.assign(risk_ratio=risk_ratios)
        itemsets_df = itemsets_df.assign(cardinality=n_items)

        # sort by descending relative risk ratio, outlier support and cardinality
        itemsets_df = itemsets_df.sort_values(
            by=["risk_ratio", "support", "cardinality"], ascending=[False, False, False]
        )

        # important features are set to those in the itemset with best risk ratio, then support, then cardinality
        important_itemset = itemsets_df.iloc[0]["itemsets"]
        feature_to_importance = dict()
        feature_to_intervals = dict()
        for item in important_itemset:
            ft, bin_idx = [int(v) for v in item.split("_")]
            feature_to_importance[ft] = 1.0
            s = features_bins[ft][bin_idx]
            e = features_bins[ft][bin_idx + 1]
            feature_to_intervals[ft] = [(s, e, True, False)]
        # return the important features and corresponding anomalous intervals
        return {
            "feature_to_importance": feature_to_importance,
            "feature_to_intervals": feature_to_intervals,
        }

    def get_boolean_transactions(
        self, normal_records: np.array, anomalous_records: np.array
    ) -> (pd.DataFrame, pd.DataFrame, list):
        """Returns the normal and anomalous boolean transactions corresponding to the provided records.

        Normal and anomalous transactions are returned as boolean DataFrames, where a value
        corresponds to the presence/absence of the corresponding item in the transaction.

        When building the boolean transactions, discrete items are:
        - created from an histogram-based binning of normal record values.
        - filtered so as to only keep those of sufficient support and relative risk ratio.

        Note: In this single-item case, we set the minimum number of occurrences of an itemset
        to the average bin count (i.e., average height) of the histogram of anomalous values.
        => the minimum support attribute is therefore not used here, but only in the multi-item
        itemset filtering performed later.

        Args:
            normal_records: "normal" records of shape `(n_normal_records, n_features)`.
            anomalous_records: "anomalous records" of shape `(n_anomalous_records, n_features)`.

        Returns:
            Normal and anomalous transaction DataFrames, along with the `bins` for each feature,
              where `bin_idx` corresponds to (bins[bin_idx], bins[bin_idx+1])`, with included start
              and excluded end.
        """
        type_to_transactions = {
            "normal": pd.DataFrame(),
            "anomalous": pd.DataFrame(),
        }
        features_bins = []
        n_normal_records = len(normal_records)
        n_anomalous_records = len(anomalous_records)
        # for initial items, we set the minimum count as the average height of the anomalous histogram
        min_ft_count = max(n_anomalous_records // self.n_bins, 1)
        for ft in range(normal_records.shape[1]):
            # extract univariate time series for the feature
            normal_values = normal_records[:, ft]
            anomalous_values = anomalous_records[:, ft]

            # perform histogram-based binning and counting of the continuous feature
            normal_counts, normal_bins = np.histogram(normal_values, bins=self.n_bins)
            normal_bins[0] = -np.inf
            normal_bins[-1] = np.inf
            features_bins.append(normal_bins)
            anomalous_counts, _ = np.histogram(anomalous_values, bins=normal_bins)

            # compute relative risk ratio for each bin of sufficient support
            bin_ids = np.where(anomalous_counts >= min_ft_count)[0]
            # number of occurrences of the range within outliers and inliers
            a_o = anomalous_counts[bin_ids]
            a_i = normal_counts[bin_ids]
            # number of non-occurrences of the range within outliers and inliers
            b_o = n_anomalous_records - a_o
            b_i = n_normal_records - a_i
            # total number of occurrences and non-occurrences of the range
            a = a_o + a_i
            b = b_o + b_i
            a[a == 0] = 1
            b[b == 0] = 1
            b_o[b_o == 0] = 1
            risk_ratios = np.nan_to_num((a_o / a) / (b_o / b))

            # only keep bins of sufficient relative risk
            for i, bin_id in zip(range(len(risk_ratios)), bin_ids):
                if risk_ratios[i] >= self.min_risk_ratio:
                    for k, v in zip(
                        ["normal", "anomalous"], [normal_values, anomalous_values]
                    ):
                        type_to_transactions[k] = type_to_transactions[k].assign(
                            **{
                                f"{ft}_{bin_id}": (v >= normal_bins[bin_id])
                                & (v < normal_bins[bin_id + 1])
                            }
                        )
        return (
            type_to_transactions["normal"],
            type_to_transactions["anomalous"],
            features_bins,
        )

    @staticmethod
    def get_itemset_risk_ratio(
        itemset: frozenset,
        normal_transactions: pd.DataFrame,
        anomalous_transactions: pd.DataFrame,
    ) -> float:
        """Returns the relative risk ratio of the provided `itemset` based on the transactions.

        Args:
            itemset: itemset for which to compute the relative risk ratio.
            normal_transactions: boolean DataFrame of normal transactions.
            anomalous_transactions: boolean DataFrame of anomalous transactions.

        Returns:
            Relative risk ratio of the itemset.
        """
        n_normal_records = len(normal_transactions)
        n_anomalous_records = len(anomalous_transactions)
        normal_occurrences = normal_transactions[list(itemset)].all(axis=1)
        anomalous_occurrences = anomalous_transactions[list(itemset)].all(axis=1)
        # number of occurrences of the itemset within outliers and inliers
        a_o = anomalous_occurrences.sum()
        a_i = normal_occurrences.sum()
        # number of non-occurrences of the itemset within outliers and inliers
        b_o = n_anomalous_records - a_o
        b_i = n_normal_records - a_i
        # total number of occurrences and non-occurrences of the itemset
        a = a_o + a_i
        b = b_o + b_i
        if a == 0:
            a = 1
        if b == 0:
            b = 1
        if b_o == 0:
            b_o = 1
        # relative risk ratio
        return np.nan_to_num((a_o / a) / (b_o / b))
