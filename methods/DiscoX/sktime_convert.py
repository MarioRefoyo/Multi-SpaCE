'''This file is written by the authors of the sktime library. We only made some modifications
to resolve some errors.'''

import numpy as np
import pandas as pd

__all__ = [
    "convert_dict",
]

__all__ = [
    "MTYPE_REGISTER_PANEL",
    "MTYPE_LIST_PANEL",
]


MTYPE_REGISTER_PANEL = [
    (
        "nested_univ",
        "Panel",
        "pd.DataFrame with one column per variable, pd.Series in cells",
    ),
    (
        "numpy3D",
        "Panel",
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
    ),
    (
        "numpyflat",
        "Panel",
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
    ),
    ("pd-multiindex", "Panel", "pd.DataFrame with multi-index (instance, time point)"),
    ("pd-wide", "Panel", "pd.DataFrame in wide format, cols = (instance*time point)"),
    (
        "pd-long",
        "Panel",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
    ),
    ("df-list", "Panel", "list of pd.DataFrame"),
]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values


# dictionary indexed by triples of types
#  1st element = convert from - type
#  2nd element = convert to - type
#  3rd element = considered as this scitype - string
# elements are conversion functions of machine type (1st) -> 2nd

convert_dict = dict()


def convert_identity(what, store=None):

    return what


# assign identity function to type conversion to self
for tp in MTYPE_LIST_PANEL:
    convert_dict[(tp, tp, "Panel")] = convert_identity


def _cell_is_series_or_array(cell):
    return isinstance(cell, (pd.Series, np.ndarray))


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series_or_array)


def are_columns_nested(X):
    """Check whether any cells have nested structure in each DataFrame column.
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for nested data structures.
    Returns
    -------
    any_nested : bool
        If True, at least one column is nested.
        If False, no nested columns.
    """
    any_nested = _nested_cell_mask(X).any().values
    return any_nested


def _nested_cell_timepoints(cell):
    if _cell_is_series_or_array(cell):
        n_timepoints = cell.shape[0]
    else:
        n_timepoints = 0
    return n_timepoints


def _check_equal_index(X):
    """Check if all time-series in nested pandas DataFrame have the same index.
    Parameters
    ----------
    X : nested pd.DataFrame
        Input dataframe with time-series in cells.
    Returns
    -------
    indexes : list of indixes
        List of indixes with one index for each column
    """
    # TODO handle 1d series, not only 2d dataframes
    # TODO assumes columns are typed (i.e. all rows for a given column have
    #  the same type)
    # TODO only handles series columns, raises error for columns with
    #  primitives

    indexes = []
    # Check index for each column separately.
    for c, col in enumerate(X.columns):

        # Get index from first row, can be either pd.Series or np.array.
        first_index = (
            X.iloc[0, c].index
            if hasattr(X.iloc[0, c], "index")
            else np.arange(X.iloc[c, 0].shape[0])
        )

        # Series must contain at least 2 observations, otherwise should be
        # primitive.
        if len(first_index) < 2:
            raise ValueError(
                f"Time series must contain at least 2 observations, "
                f"but found: "
                f"{len(first_index)} observations in column: {col}"
            )

        # Check index for all rows.
        for i in range(1, X.shape[0]):
            index = (
                X.iloc[i, c].index
                if hasattr(X.iloc[i, c], "index")
                else np.arange(X.iloc[c, 0].shape[0])
            )
            if not np.array_equal(first_index, index):
                raise ValueError(
                    f"Found time series with unequal index in column {col}. "
                    f"Input time-series must have the same index."
                )
        indexes.append(first_index)

    return indexes


def from_3d_numpy_to_2d_array(X):
    """Convert 2D NumPy Panel to 2D numpy Panel.
    Converts 3D numpy array (n_instances, n_columns, n_timepoints) to
    a 2D numpy array with shape (n_instances, n_columns*n_timepoints)
    Parameters
    ----------
    X : np.ndarray
        The input 3d-NumPy array with shape
        (n_instances, n_columns, n_timepoints)
    Returns
    -------
    array_2d : np.ndarray
        A 2d-NumPy array with shape (n_instances, n_columns*n_timepoints)
    """
    array_2d = X.reshape(X.shape[0], -1)
    return array_2d


def from_3d_numpy_to_2d_array_adp(what, store=None):

    return from_3d_numpy_to_2d_array(what)


convert_dict[("numpy3D", "numpyflat", "Panel")] = from_3d_numpy_to_2d_array_adp


def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested Panel to 2D numpy Panel.
    Convert nested pandas DataFrame or Series with NumPy arrays or
    pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the
    same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires
    series to be have the same index.
    Parameters
    ----------
    X : nested pd.DataFrame or nested pd.Series
    return_numpy : bool, default = False
        - If True, returns a NumPy array of the tabular data.
        - If False, returns a pandas DataFrame with row and column names.
    Returns
    -------
     Xt : pandas DataFrame
        Transformed DataFrame in tabular format
    """
    # TODO does not handle dataframes with nested series columns *and*
    #  standard columns containing only primitives

    # convert nested data into tabular data
    if isinstance(X, pd.Series):
        Xt = np.array(X.tolist())

    elif isinstance(X, pd.DataFrame):
        try:
            Xt = np.hstack([X.iloc[:, i].tolist() for i in range(X.shape[1])])

        # except strange key error for specific case
        except KeyError:
            if (X.shape == (1, 1)) and (X.iloc[0, 0].shape == (1,)):
                # in fact only breaks when an additional condition is met,
                # namely that the index of the time series of a single value
                # does not start with 0, e.g. pd.RangeIndex(9, 10) as is the
                # case in forecasting
                Xt = X.iloc[0, 0].values
            else:
                raise

        if Xt.ndim != 2:
            raise ValueError(
                "Tabularization failed, it's possible that not "
                "all series were of equal length"
            )

    else:
        raise ValueError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )

    if return_numpy:
        return Xt

    Xt = pd.DataFrame(Xt)

    # create column names from time index
    if X.ndim == 1:
        time_index = (
            X.iloc[0].index
            if hasattr(X.iloc[0], "index")
            else np.arange(X.iloc[0].shape[0])
        )
        columns = [f"{X.name}__{i}" for i in time_index]

    else:
        columns = []
        for colname, col in X.items():
            time_index = (
                col.iloc[0].index
                if hasattr(col.iloc[0], "index")
                else np.arange(col.iloc[0].shape[0])
            )
            columns.extend([f"{colname}__{i}" for i in time_index])

    Xt.index = X.index
    Xt.columns = columns
    return Xt


def from_nested_to_pdwide(what, store=None):

    return from_nested_to_2d_array(X=what, return_numpy=False)


def from_nested_to_2d_np_array(what, store=None):

    return from_nested_to_2d_array(X=what, return_numpy=True)


convert_dict[("nested_univ", "pd-wide", "Panel")] = from_nested_to_pdwide

convert_dict[("nested_univ", "numpyflat", "Panel")] = from_nested_to_2d_np_array


def from_2d_array_to_nested(
    X, index=None, columns=None, time_index=None, cells_as_numpy=False
):
    """Convert 2D dataframe to nested dataframe.
    Convert tabular pandas DataFrame with only primitives in cells into
    nested pandas DataFrame with a single column.
    Parameters
    ----------
    X : pd.DataFrame
    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series
    index : array-like, shape=[n_samples], optional (default = None)
        Sample (row) index of transformed DataFrame
    time_index : array-like, shape=[n_obs], optional (default = None)
        Time series index of transformed DataFrame
    Returns
    -------
    Xt : pd.DataFrame
        Transformed DataFrame in nested format
    """
    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series

    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt


def from_pd_wide_to_nested(what, store=None):

    return from_2d_array_to_nested(X=what)


convert_dict[("pd-wide", "nested_univ", "Panel")] = from_pd_wide_to_nested


def convert_from_dictionary(ts_dict):
    """Conversion from dictionary to pandas.
    Simple conversion from a dictionary of each series, e.g. univariate
        x = {
            "Series1": [1.0,2.0,3.0,1.0,2.0],
            "Series2": [3.0,2.0,1.0,3.0,2.0],
        }
    or multivariate, e.g.
    to sktime pandas format
    TODO: Adapt for multivariate
    """
    panda = pd.DataFrame(ts_dict)
    panda = panda.transpose()
    return from_2d_array_to_nested(panda)


def _concat_nested_arrays(arrs, cells_as_numpy=False):
    """Nest tabular arrays from nested list.
    Helper function to nest tabular arrays from nested list of arrays.
    Parameters
    ----------
    arrs : list of numpy arrays
        Arrays must have the same number of rows, but can have varying
        number of columns.
    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series
    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe with nested column for each input array.
    """
    if cells_as_numpy:
        Xt = pd.DataFrame(
            np.column_stack(
                [pd.Series([np.array(vals) for vals in interval]) for interval in arrs]
            )
        )
    else:
        Xt = pd.DataFrame(
            np.column_stack(
                [pd.Series([pd.Series(vals) for vals in interval]) for interval in arrs]
            )
        )
    return Xt


def _get_index(x):
    if hasattr(x, "index"):
        return x.index
    else:
        # select last dimension for time index
        return pd.RangeIndex(x.shape[-1])


def _get_time_index(X):
    """Get index of time series data, helper function.
    Parameters
    ----------
    X : pd.DataFrame
    Returns
    -------
    time_index : pandas Index
        Index of time series
    """
    # assumes that all samples share the same the time index, only looks at
    # first row
    if isinstance(X, pd.DataFrame):
        return _get_index(X.iloc[0, 0])

    elif isinstance(X, pd.Series):
        return _get_index(X.iloc[0])

    elif isinstance(X, np.ndarray):
        return _get_index(X)

    else:
        raise ValueError(
            f"X must be a pandas DataFrame or Series, but found: {type(X)}"
        )


def _make_column_names(column_count):
    return [f"var_{i}" for i in range(column_count)]


def _get_column_names(X):
    if isinstance(X, pd.DataFrame):
        return X.columns
    else:
        return _make_column_names(X.shape[1])


def from_nested_to_long(
    X, instance_column_name=None, time_column_name=None, dimension_column_name=None
):
    """Convert nested DataFrame to long DataFrame.
    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame
    instance_column_name : str
        The name of column corresponding to the DataFrame's instances
    time_column_name : str
        The name of the column corresponding to the DataFrame's timepoints.
    dimension_column_name : str
        The name of the column corresponding to the DataFrame's dimensions.
    Returns
    -------
    long_df : pd.DataFrame
        Long pandas DataFrame
    """
    long_df = from_nested_to_multi_index(
        X, instance_index="index", time_index="time_index"
    )
    long_df.reset_index(inplace=True)
    long_df = long_df.melt(id_vars=["index", "time_index"], var_name="column")

    col_rename_dict = {}
    if instance_column_name is not None:
        col_rename_dict["index"] = instance_column_name

    if time_column_name is not None:
        col_rename_dict["time_index"] = time_column_name

    if dimension_column_name is not None:
        col_rename_dict["column"] = dimension_column_name

    if len(col_rename_dict) > 0:
        long_df = long_df.rename(columns=col_rename_dict)

    return long_df


def from_nested_to_long_adp(what, store=None):

    return from_nested_to_long(
        X=what,
        instance_column_name="case_id",
        time_column_name="reading_id",
        dimension_column_name="dim_id",
    )


convert_dict[("nested_univ", "pd-long", "Panel")] = from_nested_to_long_adp


def from_long_to_nested(
    X_long,
    instance_column_name="case_id",
    time_column_name="reading_id",
    dimension_column_name="dim_id",
    value_column_name="value",
    column_names=None,
):
    """Convert long DataFrame to a nested DataFrame.
    Parameters
    ----------
    X_long : pd.DataFrame
        The long DataFrame
    instance_column_name : str, default = 'case_id'
        The name of column corresponding to the DataFrame's instances.
    time_column_name : str, default = 'reading_id'
        The name of the column corresponding to the DataFrame's timepoints.
    dimension_column_name : str, default = 'dim_id'
        The name of the column corresponding to the DataFrame's dimensions.
    value_column_name : str, default = 'value'
        The name of the column corresponding to the DataFrame's values.
    column_names : list, optional
        Optional list of column names to use for nested DataFrame columns.
    Returns
    -------
    X_nested : pd.DataFrame
        Nested pandas DataFrame
    """
    X_nested = X_long.pivot(
        index=[instance_column_name, time_column_name],
        columns=dimension_column_name,
        values=value_column_name,
    )
    X_nested = from_multi_index_to_nested(X_nested, instance_index=instance_column_name)

    n_columns = X_nested.shape[1]
    if column_names is None:
        X_nested.columns = _make_column_names(n_columns)

    else:
        X_nested.columns = column_names

    # # get distinct dimension ids
    # unique_dim_ids = long_dataframe.iloc[:, 1].unique()
    # num_dims = len(unique_dim_ids)

    # data_by_dim = []
    # indices = []

    # # get number of distinct cases (note: a case may have 1 or many dimensions)
    # unique_case_ids = long_dataframe.iloc[:, 0].unique()
    # # assume series are indexed from 0 to m-1 (can map to non-linear indices
    # # later if needed)

    # # init a list of size m for each d - to store the series data for m
    # # cases over d dimensions
    # # also, data may not be in order in long format so store index data for
    # # aligning output later
    # # (i.e. two stores required: one for reading id/timestamp and one for
    # # value)
    # for d in range(0, num_dims):
    #     data_by_dim.append([])
    #     indices.append([])
    #     for _c in range(0, len(unique_case_ids)):
    #         data_by_dim[d].append([])
    #         indices[d].append([])

    # # go through every row in the dataframe
    # for i in range(0, len(long_dataframe)):
    #     # extract the relevant data, catch cases where the dim id is not an
    #     # int as it must be the class

    #     row = long_dataframe.iloc[i]
    #     case_id = int(row[0])
    #     dim_id = int(row[1])
    #     reading_id = int(row[2])
    #     value = row[3]
    #     data_by_dim[dim_id][case_id].append(value)
    #     indices[dim_id][case_id].append(reading_id)

    # x_data = {}
    # for d in range(0, num_dims):
    #     key = "dim_" + str(d)
    #     dim_list = []
    #     for i in range(0, len(unique_case_ids)):
    #         temp = pd.Series(data_by_dim[d][i], indices[d][i])
    #         dim_list.append(temp)
    #     x_data[key] = pd.Series(dim_list)

    return X_nested


def from_long_to_nested_adp(what, store=None):

    return from_long_to_nested(X_long=what)


convert_dict[("pd-long", "nested_univ", "Panel")] = from_nested_to_long_adp


def from_multi_index_to_3d_numpy(X, instance_index=None, time_index=None):
    """Convert pandas multi-index Panel to numpy 3D Panel.
    Convert panel data stored as pandas multi-index DataFrame to
    Numpy 3-dimensional NumPy array (n_instances, n_columns, n_timepoints).
    Parameters
    ----------
    X : pd.DataFrame
        The multi-index pandas DataFrame
    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances
    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints
    Returns
    -------
    X_3d : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    """
    if X.index.nlevels != 2:
        raise ValueError("Multi-index DataFrame should have 2 levels.")

    if (instance_index is None) or (time_index is None):
        msg = "Must supply parameters instance_index and time_index"
        raise ValueError(msg)

    n_instances = len(X.groupby(level=instance_index))
    # Alternative approach is more verbose
    # n_instances = (multi_ind_dataframe
    #                    .index
    #                    .get_level_values(instance_index)
    #                    .unique()).shape[0]
    n_timepoints = len(X.groupby(level=time_index))
    # Alternative approach is more verbose
    # n_instances = (multi_ind_dataframe
    #                    .index
    #                    .get_level_values(time_index)
    #                    .unique()).shape[0]

    n_columns = X.shape[1]

    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)

    return X_3d


def from_multi_index_to_3d_numpy_adp(what, store=None):

    return from_multi_index_to_3d_numpy(
        X=what, instance_index="instances", time_index="timepoints"
    )


convert_dict[("pd-multiindex", "numpy3D", "Panel")] = from_multi_index_to_3d_numpy_adp


def from_3d_numpy_to_multi_index(
    X, instance_index=None, time_index=None, column_names=None
):
    """Convert 3D numpy Panel to pandas multi-index Panel.
    Convert 3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    to panel data stored as pandas multi-indexed DataFrame.
    Parameters
    ----------
    X : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances
    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints
    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame
    """
    if X.ndim != 3:
        msg = " ".join(
            [
                "Input should be 3-dimensional NumPy array with shape",
                "(n_instances, n_columns, n_timepoints).",
            ]
        )
        raise TypeError(msg)

    n_instances, n_columns, n_timepoints = X.shape
    multi_index = pd.MultiIndex.from_product(
        [range(n_instances), range(n_columns), range(n_timepoints)],
        names=["node_id", "columns", "timestamp"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")

    # Assign column names
    if column_names is None:
        X_mi.columns = _make_column_names(n_columns)

    else:
        X_mi.columns = column_names

    index_rename_dict = {}
    if instance_index is not None:
        index_rename_dict["node_id"] = instance_index

    if time_index is not None:
        index_rename_dict["timestamp"] = time_index

    if len(index_rename_dict) > 0:
        X_mi = X_mi.rename_axis(index=index_rename_dict)

    return X_mi


def from_3d_numpy_to_multi_index_adp(what, store=None):

    return from_3d_numpy_to_multi_index(X=what)


convert_dict[("numpy3D", "pd-multiindex", "Panel")] = from_3d_numpy_to_multi_index_adp


def from_multi_index_to_nested(
    multi_ind_dataframe, instance_index=None, cells_as_numpy=False
):
    """Convert a pandas DataFrame witha multi-index to a nested DataFrame.
    Parameters
    ----------
    multi_ind_dataframe : pd.DataFrame
        Input multi-indexed pandas DataFrame
    instance_index_name : str
        The name of multi-index level corresponding to the DataFrame's instances
    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series
    Returns
    -------
    x_nested : pd.DataFrame
        The nested version of the DataFrame
    """
    if instance_index is None:
        raise ValueError("Supply a value for the instance_index parameter.")

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    instance_idxs = multi_ind_dataframe.index.get_level_values(instance_index).unique()

    x_nested = pd.DataFrame()

    # Loop the dimensions (columns) of multi-index DataFrame
    for _label, _series in multi_ind_dataframe.iteritems():  # noqa
        # for _label in multi_ind_dataframe.columns:
        #    _series = multi_ind_dataframe.loc[:, _label]
        # Slice along the instance dimension to return list of series for each case
        # Note: if you omit .rename_axis the returned DataFrame
        #       prints time axis dimension at the start of each cell,
        #       but this doesn't affect the values.
        if cells_as_numpy:
            dim_list = [
                _series.xs(instance_idx, level=instance_index).values
                for instance_idx in instance_idxs
            ]
        else:
            dim_list = [
                _series.xs(instance_idx, level=instance_index).rename_axis(None)
                for instance_idx in instance_idxs
            ]

        x_nested[_label] = pd.Series(dim_list)
    x_nested = pd.DataFrame(x_nested)

    col_msg = "Multi-index and nested DataFrames should have same columns names"
    assert (x_nested.columns == multi_ind_dataframe.columns).all(), col_msg

    return x_nested


def from_multi_index_to_nested_adp(what, store=None):

    return from_multi_index_to_nested(X=what, instance_index_name="instances")


convert_dict[("pd-multiindex", "nested_univ", "Panel")] = from_multi_index_to_nested_adp


def from_nested_to_multi_index(X, instance_index=None, time_index=None):
    """Convert nested pandas Panel to multi-index pandas Panel.
    Converts nested pandas DataFrame (with time series as pandas Series
    or NumPy array in cells) into multi-indexed pandas DataFrame.
    Can convert mixed nested and primitive DataFrame to multi-index DataFrame.
    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame to convert to a multi-indexed pandas DataFrame
    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances
    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints
    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame
    """
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    if time_index is None:
        time_index_name = "timestamp"
    else:
        time_index_name = time_index

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    if instance_index is None:
        instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = "node_id"

    else:
        if instance_index in X.index.names:
            instance_idxs = X.index.get_level_values(instance_index).unique()
        else:
            instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = instance_index

    instances = []
    for instance_idx in instance_idxs:
        instance = [
            _val if isinstance(_val, pd.Series) else pd.Series(_val, name=_lab)
            for _lab, _val in X.loc[instance_idx, :].iteritems()  # noqa
        ]
        # instance = [
        #     X.loc[instance_idx, _label]
        #     if isinstance(X.loc[instance_idx, _label], pd.Series)
        #     else pd.Series(X.loc[instance_idx, _label], name=_label)
        #     for _label in X.columns ]

        instance = pd.concat(instance, axis=1)
        # For primitive (non-nested column) assume the same
        # primitive value applies to every timepoint of the instance
        for col_idx, is_nested in enumerate(nested_col_mask):
            if not is_nested:
                instance.iloc[:, col_idx] = instance.iloc[:, col_idx].ffill()

        # Correctly assign multi-index
        multi_index = pd.MultiIndex.from_product(
            [[instance_idx], instance.index],
            names=[instance_index_name, time_index_name],
        )
        instance.index = multi_index
        instances.append(instance)

    X_mi = pd.concat(instances)
    X_mi.columns = X.columns

    return X_mi


def from_nested_to_multi_index_adp(what, store=None):

    return from_nested_to_multi_index(
        X=what, instance_index_name="instances", time_index="timepoints"
    )


convert_dict[("nested_univ", "pd-multiindex", "Panel")] = from_nested_to_multi_index_adp


def _convert_series_cell_to_numpy(cell):
    if isinstance(cell, pd.Series):
        return cell.to_numpy()
    else:
        return cell


def from_nested_to_3d_numpy(X):
    """Convert nested Panel to 3D numpy Panel.
    Convert nested pandas DataFrame (with time series as pandas Series
    in cells) into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).
    Parameters
    ----------
    X : pd.DataFrame
        Nested pandas DataFrame
    Returns
    -------
    X_3d : np.ndarrray
        3-dimensional NumPy array
    """
    # n_instances, n_columns = X.shape
    # n_timepoints = X.iloc[0, 0].shape[0]
    # array = np.empty((n_instances, n_columns, n_timepoints))
    # for column in range(n_columns):
    #     array[:, column, :] = X.iloc[:, column].tolist()
    # return array
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    # If all the columns are nested in structure
    if nested_col_mask.count(True) == len(nested_col_mask):
        X_3d = np.stack(
            X.applymap(_convert_series_cell_to_numpy)
            .apply(lambda row: np.stack(row), axis=1)
            .to_numpy()
        )

    # If some columns are primitive (non-nested) then first convert to
    # multi-indexed DataFrame where the same value of these columns is
    # repeated for each timepoint
    # Then the multi-indexed DataFrame can be converted to 3d NumPy array
    else:
        X_mi = from_nested_to_multi_index(X)
        X_3d = from_multi_index_to_3d_numpy(
            X_mi, instance_index="instance", time_index="timepoints"
        )

    return X_3d


def from_nested_to_3d_numpy_adp(what, store=None):

    return from_nested_to_3d_numpy(X=what)


convert_dict[("nested_univ", "numpy3D", "Panel")] = from_nested_to_3d_numpy_adp


def from_3d_numpy_to_nested(X, column_names=None, cells_as_numpy=False):
    """Convert numpy 3D Panel to nested pandas Panel.
    Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into nested pandas DataFrame (with time series as pandas Series in cells)
    Parameters
    ----------
    X : np.ndarray
        3-dimensional Numpy array to convert to nested pandas DataFrame format
    column_names: list-like, default = None
        Optional list of names to use for naming nested DataFrame's columns
    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series
    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.DataFrame()
    # n_instances, n_variables, _ = X.shape
    n_instances, n_columns, n_timepoints = X.shape

    container = np.array if cells_as_numpy else pd.Series

    if column_names is None:
        column_names = _make_column_names(n_columns)

    else:
        if len(column_names) != n_columns:
            msg = " ".join(
                [
                    f"Input 3d Numpy array as {n_columns} columns,",
                    f"but only {len(column_names)} names supplied",
                ]
            )
            raise ValueError(msg)

    for j, column in enumerate(column_names):
        df[column] = [container(X[instance, j, :]) for instance in range(n_instances)]
    return df


def from_3d_numpy_to_nested_adp(what, store=None):

    return from_3d_numpy_to_nested(X=what)


convert_dict[("numpy3D", "nested_univ", "Panel")] = from_3d_numpy_to_nested_adp


def is_nested_dataframe(X):
    """Check whether the input is a nested DataFrame.
    To allow for a mixture of nested and primitive columns types the
    the considers whether any column is a nested np.ndarray or pd.Series.
    Column is consider nested if any cells in column have a nested structure.
    Parameters
    ----------
    X: Input that is checked to determine if it is a nested DataFrame.
    Returns
    -------
    bool: Whether the input is a nested DataFrame
    """
    # return isinstance(X, pd.DataFrame) and isinstance(
    #     X.iloc[0, 0], (np.ndarray, pd.Series)
    # )
    is_dataframe = isinstance(X, pd.DataFrame)

    # If not a DataFrame we know is_nested_dataframe is False
    if not is_dataframe:
        return is_dataframe

    # Otherwise we'll see if any column has a nested structure in first row
    else:
        is_nested = are_columns_nested(X).any()

        return is_dataframe and is_nested
    
def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    # Parse the file
    with open(full_file_path_and_name, encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError(
                            "timestamps tag requires an associated Boolean " "value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise OSError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError(
                            "univariate tag requires an associated Boolean  " "value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise OSError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError(
                            "classlabel tag requires an associated Boolean  " "value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise OSError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    if data_started:
                        raise OSError("metadata must come before data")
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError(
                            "targetlabel tag requires an associated Boolean value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid targetlabel value")
                    if token_len > 2:
                        raise OSError(
                            "targetlabel tag should not be accompanied with info "
                            "apart from true/false, but found "
                            f"{tokens}"
                        )
                    has_class_labels_tag = True
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise OSError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise OSError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise OSError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we are dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if class_val not in class_label_list:
                                            raise OSError(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise OSError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise OSError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise OSError(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise OSError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise OSError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise OSError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise OSError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise OSError(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise OSError("metadata incomplete")

        elif metadata_started and not data_started:
            raise OSError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise OSError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise OSError("empty file")
