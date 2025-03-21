from abc import abstractmethod

import pandas as pd
from tqdm import tqdm


class BaseCrafter:
    """Base feature engineering/crafting class.

    Features crafting steps are typically gathered into what we call "alteration bundles".

    We start with an empty "result" DataFrame for each period, and sequentially add columns to it based on the
    steps described in the considered bundle.

    A bundle step is of the form `input_features: alteration_chain`. It projects the original period
    DataFrame on `input_features` only, applies `alteration_chain` to them and adds the output columns
    to the result DataFrame.

    `input_features` (tuple|str): Either a tuple of feature names or "all" if we want to consider
        all the original input features.

    `alteration_chain` (str): dot-separated alteration functions to apply to the input features, where
        the output columns of a function constitute the input to the next.
        Each alteration function must be of the form `fname_arg1_arg2_..._argn`, where
        - `fname` refers to the function name, as defined in the `self.alteration_functions` dictionary.
            If the name consists of multiple words, they should typically be merged without spaces.
        - `{argi}` are the function's arguments, underscore-separated and whose number/order have to match
            the function definition.

    Note: alteration functions do not assume any "anomaly" column is present in the input DataFrames.

    Args:
        bundle_idx: alteration bundle index in `bundles_list`.
    """

    @property
    @abstractmethod
    def bundles_list(self):
        """List of possible alteration bundles (i.e., "feature sets")."""

    def __init__(self, bundle_idx: int = 0):
        # feature alteration bundle
        self.alteration_bundle = self.bundles_list[bundle_idx]
        # mapping from names to feature alteration functions
        self.alteration_functions = dict(
            {"identity": apply_identity, "difference": add_differencing},
            **self.get_alteration_functions_extension(),
        )

    @abstractmethod
    def get_alteration_functions_extension(self):
        """Returns a dictionary to append to `self.alteration_functions`.

        This dictionary can either be empty or contain any data-specific alteration functions.
        """

    def get_altered_features(self, datasets):
        """Returns the provided datasets with their features altered by `self.alteration_bundle`.

        Args:
            datasets (dict): datasets of the form `{set_name: period_dfs}`.

        Returns:
            dict: the datasets with altered features, in the same format as provided.
        """
        altered_datasets = dict()
        for set_name in datasets:
            print(f"altering features of {set_name} periods:")
            altered_datasets[set_name] = []
            # sequentially apply the alteration bundle to all periods of the dataset
            for period_df in tqdm(datasets[set_name]):
                altered_datasets[set_name].append(
                    self.apply_alteration_bundle(period_df)
                )
        return altered_datasets

    def apply_alteration_bundle(self, period_df):
        """Returns `period_df` with its features altered by `self.alteration_bundle`.

        Args:
            period_df (pd.DataFrame): input period DataFrame whose features to alter.

        Returns:
            pd.DataFrame: the same period with its features altered by `self.alteration_bundle`.
        """
        # if any, remove the period's "anomaly" column from consideration
        input_df = period_df.copy()
        if "anomaly" in period_df:
            input_df = period_df.drop("anomaly", axis=1)

        # sequentially add the outputs of the bundle's alteration steps to an empty result DataFrame
        result_df = pd.DataFrame()
        for input_features, alteration_chain in self.alteration_bundle.items():
            # consider all features if `input_features` is `all`
            input_features = (
                slice(None) if input_features == "all" else list(input_features)
            )
            # constitute the chain's output by sequentially calling its functions with their arguments
            chain_output_df = input_df[input_features]
            for alteration_step in alteration_chain.split("."):
                alteration_specs = alteration_step.split("_")
                chain_output_df = self.alteration_functions[alteration_specs[0]](
                    chain_output_df, *alteration_specs[1:]
                )
            # add the chain output to the period (filling missing rows forwards, then backwards)
            # TODO: replace `assign` by a more efficient method (pandas `PerformanceWarning`)
            result_df = result_df.assign(**chain_output_df).ffill().bfill()

        # add back any "anomaly" column to the records that were not dropped in the process (implicit join)
        if "anomaly" in period_df:
            result_df["anomaly"] = period_df["anomaly"]
        return result_df


def apply_identity(period_df):
    """Simply returns `period_df` without altering its features."""
    return period_df


def add_differencing(period_df, diff_factor_str, original_treatment):
    """Adds features differences, either keeping or dropping the original ones.

    Args:
        period_df (pd.DataFrame): input period DataFrame.
        diff_factor_str (str): differencing factor as a string integer.
        original_treatment (str): either `keep` or `drop`, specifying what to do with original features.

    Returns:
        pd.DataFrame: the input DataFrame with differenced features, with or without the original ones.
    """
    assert original_treatment in [
        "drop",
        "keep",
    ], "original features treatment can only be `keep` or `drop`"
    # apply differencing and drop records with NaN values
    difference_df = period_df.diff(int(diff_factor_str)).dropna()
    difference_df.columns = [
        f"{c}_{diff_factor_str}_diff" for c in difference_df.columns
    ]
    # prepend original input features if we choose to keep them (implicit join if different counts)
    if original_treatment == "keep":
        difference_df = pd.concat([period_df, difference_df], axis=1)
    return difference_df
