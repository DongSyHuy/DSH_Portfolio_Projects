"""Functions for Data Profilling, Data Preparation"""

import os, glob, sys, time, re, math
from IPython.display import display
import unicodedata
import numpy as np
import pandas as pd
import pingouin as pg
from thefuzz import process


#
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Timing functions
def timing__decorator(func):
    """
    https://towardsdatascience.com/python-decorators-for-data-science-6913f717669a
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = round((end_time - start_time), 2)
        (
            print(f"\nExecution time: {running_time} sec.\n")
            if running_time < 60
            else print(
                f"\nExecution time:  {running_time//60} min and {math.ceil(running_time%60)} sec.\n"
            )
        )
        return result

    return wrapper


#
def my_folder_path(
    folder_to_list: str = None, print_list=True, folder_for_path: str = None
) -> str:
    """
    folder_to_list: is folder in the same level with current folder.\n
    folder_for_path: is folder in the same level with current folder or sub-folder of folder_to_list
    """
    mypath = ""
    parent_folder = os.path.dirname(os.getcwd()).replace("\\", "/")
    same_lv_folders = os.listdir(parent_folder)
    notebook_folder_path = os.getcwd().replace("\\", "/")
    print(f"\nCurrent folder of this notebook: \n{notebook_folder_path}", end="\n\n")
    # if same_level_folder is None:
    notebook_folder = notebook_folder_path.split("/")[-1]
    if print_list is True:
        print(
            f"Folders and/or files in the same level with '{notebook_folder}':",
            end="\n",
        )
        for o in same_lv_folders:
            p1 = os.path.join(parent_folder, o).replace("\\", "/")
            print(p1) if p1 != notebook_folder_path else None

    if folder_to_list in same_lv_folders:
        lv1_sub_folders = os.listdir(
            os.path.join(parent_folder, folder_to_list).replace("\\", "/")
        )
        if print_list is True:
            print(f"\nSub-folders & files in '{folder_to_list}':")
            for i in lv1_sub_folders:
                p = os.path.join(parent_folder, folder_to_list, i).replace("\\", "/")
                print(p)
                files = glob.glob(os.path.join(p, "*.*"))
                # loop over the list of files
                for f in files:
                    f = f.replace("\\", "/")
                    print(f.split("/")[-1])
        if folder_for_path in same_lv_folders:
            mypath = os.path.join(parent_folder, folder_for_path) + "/"
        elif folder_for_path in lv1_sub_folders:
            mypath = os.path.join(parent_folder, folder_to_list, folder_for_path) + "/"
        elif folder_for_path is not None:
            os.mkdir(os.path.join(parent_folder, folder_to_list, folder_for_path))
            mypath = os.path.join(parent_folder, folder_to_list, folder_for_path) + "/"
        else:
            mypath = os.path.join(parent_folder, notebook_folder) + "/"
    elif folder_to_list is not None:
        os.mkdir(os.path.join(parent_folder, folder_to_list))
        mypath = my_folder_path(
            folder_to_list=folder_to_list,
            print_list=print_list,
            folder_for_path=folder_for_path,
        )
    else:
        mypath = os.path.join(parent_folder, notebook_folder) + "/"
    print("\nYour data path is: \n", mypath.replace("\\", "/"))
    return mypath.replace("\\", "/")


#
@timing__decorator
def my_read_csv(
    data_path: str,
    csv_name: str,
    seps: str,
    na_vals: list = None,
    index_col: list = None,
    parse_dates=None or list or dict,
    dtype=None or str or dict,
    header_row="infer" or None or list,
    col_name: list = None,
    clean_col_name=False,
    print_info=False,
    low_memory=True,
    strip_str_val=True,
) -> pd.DataFrame:
    """My read_csv"""
    na_valus = [
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "N/A",
        "NA",
        "n/a",
        "N/a",
        "n/A",
        "na",
        "Na",
        "nA",
        "<NA>",
        "<Na>",
        "<nA>",
        "NaN",
        "nan",
        "naN",
        "nAn",
        "NAN",
        "NULL",
        "null",
        "Null",
        "NUll",
        "nUll",
        "NaT",
        "nat",
        "None",
        "none",
        "",
    ]
    na_valus.extend(na_vals) if na_vals is not None else na_valus
    df = pd.read_csv(
        data_path + csv_name,
        sep=seps,
        parse_dates=parse_dates,
        dtype=dtype,
        na_values=na_valus,
        skipinitialspace=True,  # Strip all whitespace values from Entire DataFrame (to NaN)
        encoding="utf-8",
        encoding_errors="replace",
        header=header_row,
        names=col_name,
        index_col=index_col,
        on_bad_lines="warn",
        low_memory=low_memory,
    )

    df2 = (
        my_clean_col_name(df, clean_dict="Default", rename_dict=None, drop_col=False)
        if clean_col_name is True
        else df
    )

    if strip_str_val is True:
        for i in df2.columns:
            if df2[i].dtype in ["object", "category"]:
                df2[i] = df2[i].str.strip()
            else:
                df2[i] = df2[i]

    if print_info is True:
        print("\nDataFrame information:\n")
        print(df2.info(max_cols=200))
    else:
        print(f"\nRead {csv_name} into DataFrame: Done!")

    return df2


#
def my_check_excel(data_path, file_name):
    input = data_path + file_name
    xlsx = pd.ExcelFile(input)
    # print the list that stores the sheetnames
    print(f"List of sheets in {file_name}:\n {xlsx.sheet_names}\n")
    return xlsx.close()


#
@timing__decorator
def my_read_excel(
    data_path: str,
    file_name: str,
    sheet: str | int | list,
    header_row: int | list,
    col_name=None | list,
    clean_col_name=False,
    na_vals=None | list,
    index_col=None | list,
    dtype=None | str | dict,
    print_info=False,
) -> pd.DataFrame:
    """My read excel file"""
    na_valus = [
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "N/A",
        "NA",
        "n/a",
        "N/a",
        "n/A",
        "na",
        "Na",
        "nA",
        "<NA>",
        "<Na>",
        "<nA>",
        "NaN",
        "nan",
        "naN",
        "nAn",
        "NAN",
        "NULL",
        "null",
        "Null",
        "NUll",
        "nUll",
        "NaT",
        "nat",
        "None",
        "none",
        "",
    ]
    na_valus.extend(na_vals) if na_vals is not None else na_valus
    input = data_path + file_name
    xlsx = pd.ExcelFile(input)
    df = pd.read_excel(
        xlsx,
        sheet_name=sheet,
        header=header_row,
        names=col_name,
        index_col=index_col,
        dtype=dtype,
        na_values=na_valus,
    )
    xlsx.close()
    df2 = (
        my_clean_col_name(df, clean_dict="Default", rename_dict=None, drop_col=False)
        if clean_col_name is True
        else df.copy()
    )
    for i in df2.columns:
        if df2[i].dtype in ["object", "category"]:
            df2[i] = df2[i].str.strip()
        else:
            df2[i] = df2[i]
    (
        print(df2.info(max_cols=200))
        if print_info is True
        else print(f"Read {file_name} into DataFrame: Done!")
    )
    return df2


#
@timing__decorator
def my_read_sql(
    query: str,
    engine,
    index_col: list = None,
    parse_dates=None or list or dict,
    col_name: list = None,
    clean_col_name=False,
    print_info=False,
) -> pd.DataFrame:
    """My read_sql"""
    with engine.connect() as conn:
        df = pd.read_sql(
            sql=query,
            con=conn,
            parse_dates=parse_dates,
            columns=col_name,
            index_col=index_col,
        )
        df2 = (
            my_clean_col_name(
                df, clean_dict="Default", rename_dict=None, drop_col=False
            )
            if clean_col_name is True
            else df
        )
        for i in df2.columns:
            if df2[i].dtype in ["object", "category"]:
                df2[i] = df2[i].str.strip()
            else:
                df2[i] = df2[i]
        (
            print(df2.info(max_cols=200))
            if print_info is True
            else print(f"Read SQL query or database table into a DataFrame: Done!")
        )
        conn.close()
    return df2


#
def my_to_csv(
    df: pd.DataFrame,
    file_name: str,
    same_lv_folder_w_curr: str,
    save_to_folder: str,
    header=True or list,
    index=False,
    sep: str = ",",
    na_rep=pd.NA,
    encoding: str = "utf-8",
    errors: str = "replace",
):
    csv_file = (
        my_folder_path(
            folder_to_list=same_lv_folder_w_curr,
            print_list=False,
            folder_for_path=save_to_folder,
        )
        + f"{file_name}.csv"
    )

    df.to_csv(
        csv_file,
        header=header,
        index=index,
        sep=sep,
        na_rep=na_rep,
        encoding=encoding,
        errors=errors,
    )
    if os.path.exists(csv_file):
        print(f"\nThe file '{file_name}.csv' exists!")
        print(csv_file)


def my_clean_col_name(
    df: pd.DataFrame,
    clean_dict="Default" or dict,
    rename_dict=None or dict,
    drop_col=False or list,
):
    """
    # clean_dict="Default": substitute "" for any characters other than letters, numbers, space, _, &,-,+ and substitute "_" for space, &, -, + \n
    # rename_dict: mapper, eg: dict {"A": "a", "B": "c"} or functions (str.lower)
    """
    df2 = df.copy()
    if isinstance(clean_dict, dict):
        for k, v in clean_dict.items():
            a1 = re.compile(rf"{k}")
            for s in df2.columns:
                s2 = s
                # s2 = my_strip_ascents(s)
                s3 = a1.sub(v, s2)
                s4 = s3.strip("_").strip(" ").lower()
                df2.rename(columns=dict([(s, s4)]), inplace=True)
    elif clean_dict == "Default":
        a1 = re.compile(r"[^a-zA-Z0-9\s\_\&\-\+]+")
        a2 = re.compile(r"[\s\&\-\+]+")
        for s in df2.columns:
            s2 = s
            # s2 = my_strip_ascents(s)
            s3 = a1.sub("", s2)
            s4 = a2.sub("_", s3)
            s4 = s4.strip("_").strip(" ").lower()
            df2.rename(columns=dict([(s, s4)]), inplace=True)

    if rename_dict is not None:
        df2.rename(rename_dict, axis="columns", inplace=True)

    if isinstance(drop_col, list):
        df2.drop(columns=drop_col, inplace=True)

    return df2


#
def my_describe(
    df1: pd.DataFrame, without: list = None, cat_cutoff: int = 12
) -> pd.DataFrame:
    """
    My describe function.\n
    cat_cutoff: the cut off that a numerical variable can be treat as a categorical variable.

    """
    df = df1.drop(columns=without, inplace=False) if without is not None else df1.copy()
    # for i in df.columns:
    #     if df[i].isnull().sum() == len(df):
    #         df.drop(columns=i, inplace=True)

    top = [df[i].value_counts().idxmax() for i in df.columns]
    type = [df[i].dtypes for i in df.columns]
    zero = [
        len(df[(df[i] == 0)]) if len(df[(df[i] == 0)]) > 0 else np.nan
        for i in df.columns
    ]

    def my_is_null_sum(column):
        column2 = column.copy()
        return (
            column2.mask(column2 == "").isnull().sum()
            if column2.mask(column2 == "").isnull().sum() > 0
            else np.nan
        )

    def my_q1(column):
        return column.quantile(0.25, interpolation="linear")
        # return np.percentile(column, 25)

    def my_q3(column):
        return column.quantile(0.75, interpolation="linear")
        return np.percentile(column, 75)

    def my_5q(column):
        return column.quantile(0.05, interpolation="linear")
        return np.percentile(column, 5)

    def my_95q(column):
        return column.quantile(0.95, interpolation="linear")
        # return np.percentile(column, 95)

    def my_iqr(column):
        return my_q3(column) - my_q1(column)

    def lower(column):
        lower = (
            (my_q1(column) - 1.5 * my_iqr(column))
            if (my_q1(column) - 1.5 * my_iqr(column)) > column.min()
            else np.nan
        )
        return lower

    def upper(column):
        upper = (
            (my_q3(column) + 1.5 * my_iqr(column))
            if (my_q3(column) + 1.5 * my_iqr(column)) < column.max()
            else np.nan
        )
        return upper

    def range_df(column):
        return max(column) - min(column)

    def my_count(column):
        column2 = column.copy()
        return column2.mask(column2 == "").count()

    def my_check_distinct(column):
        return column.nunique()

    def my_most_freq(column):
        return column.value_counts().max()

    stats_list = [
        my_count,
        my_is_null_sum,
        my_check_distinct,
        # "mode",
        my_most_freq,
        "max",
        upper,
        my_95q,
        my_q3,
        "mean",
        "median",
        my_q1,
        my_5q,
        lower,
        "min",
        range_df,
        my_iqr,
        "std",
        # "var",
    ]
    stats_list2 = [
        my_count,
        my_is_null_sum,
        my_check_distinct,
        # "mode",
        my_most_freq,
    ]
    for i in df.columns:
        if my_check_distinct(df[i]) <= cat_cutoff:
            df[i] = df[i].astype(str)
    des_list = (
        (
            df[i].agg(func=stats_list2)
            if df[i].dtype in ["object", "bool", "category"]
            else df[i].agg(func=stats_list)
        )
        for i in df.columns
    )
    des = pd.concat(des_list, axis=1, ignore_index=False)
    for d in des.columns:
        if des[d].dtype == "object":
            des[d] = des[d]
        else:
            des[d] = des[d].round(2)
    des = des.T
    des.insert(0, "my_type", type)
    des.insert(3, "my_zero", zero)
    des.insert(5, "my_top", top)
    # des = des.replace([np.nan], "...").T
    des = des.infer_objects(copy=False).fillna("...").T
    des.index = (
        [
            "dtypes",
            "count",
            "empty_or_null",
            "zero",
            "distinct",
            "top",
            "freq_of_top",
            "max",
            "tukey_upper_fence",
            "95%",
            "75%",
            "mean",
            "median",
            "25%",
            "5%",
            "tukey_lower_fence",
            "min",
            "range",
            "iqr",
            "std",
            # "var",
        ]
        if len(des) > 7
        else [
            "dtypes",
            "count",
            "empty_or_null",
            "zero",
            "distinct",
            "top",
            "freq_of_top",
        ]
    )
    return des


#
# @timing__decorator
def my_check_category_data(
    df: pd.DataFrame, count=True, normalize=False, num_to_cat_cutoff: int = 12
):
    df2 = df.copy()
    for i in df2.columns:
        if df2[i].isnull().sum() == len(df2):
            df2.drop(columns=i, inplace=True)

    if num_to_cat_cutoff > 0:
        for x in df2.columns:
            if not df2[x].dtype in ["object", "bool", "category"]:
                df2[x] = (
                    df2[x].astype(str)
                    if df2[x].unique().shape[0] <= num_to_cat_cutoff
                    else df2[x]
                )
    try:
        cat_list = (
            (
                pd.DataFrame(
                    df2[i].astype(str).value_counts(dropna=False, normalize=normalize)
                ).reset_index()
                if df2[i].dtype in ["object", "bool", "category"]
                else pd.DataFrame()
            )
            for i in df2.columns
        )
        cate = pd.concat(cat_list, axis=1)
        if not cate.empty:
            if normalize is True:
                cate["proportion"] = (
                    cate["proportion"].fillna(0).map(lambda x: "{:.3%}".format(x))
                )
            else:
                cate["count"] = cate["count"].fillna(0).astype(int)
            cate.replace({"nan": "___", 0: "...", "0.000%": "..."}, inplace=True)
            cate.fillna("...", inplace=True)
            cate.rename(columns={"count": " ", "proportion": " "}, inplace=True)
            cate2 = (
                cate
                if count | normalize is True
                else cate.drop(columns=[" "], inplace=False)
            )
            print("\nUnique values in column:\n")
            return cate2
        else:
            return print("The dataframe has no category columns!")
    except KeyError:
        cat_list = (
            (
                pd.DataFrame(df2[i].astype(str).value_counts(dropna=False))
                .reset_index()
                .rename(columns={i: f"c_{i}"})
                .rename(columns={"index": i})
                if df2[i].dtype in ["object", "bool", "category"]
                else pd.DataFrame()
            )
            for i in df2.columns
        )
        cate = pd.concat(cat_list, axis=1)
        if not cate.empty:
            if count is True:
                for i in cate.columns:
                    if cate[i].dtypes != "O":
                        cate[i] = cate[i].fillna(0).astype(int)
                    else:
                        cate[i] = cate[i]
            elif normalize is True:
                for i in cate.columns:
                    if cate[i].dtypes != "O":
                        cate[i] = cate[i].fillna(0).map(lambda x: "{:.3%}".format(x))
                    else:
                        cate[i] = cate[i]
            else:
                for i in cate.columns:
                    if cate[i].dtypes != "O":
                        cate.drop(columns=i, inplace=True)
            cate.replace({"nan": "___", 0: "...", "0.000%": "..."}, inplace=True)
            cate.fillna("...", inplace=True)
            print(
                f"Category/string columns and/or numeric columns with max {num_to_cat_cutoff} unique values\n"
            )
            return cate
        else:
            return print("The dataframe has no category columns!")


#
def my_type_of_variable(df: pd.DataFrame, list_of_type: list):
    t = pd.DataFrame(df.dtypes).reset_index()
    t.rename(columns={"index": "Column", 0: "Dtype"}, inplace=True)
    (
        t.insert(
            loc=len(t.columns),
            column="Type of Variable",
            value=list_of_type,
            allow_duplicates=True,
        )
        if len(list_of_type) == len(t)
        else print("Please check the number of items in the list!")
    )
    return t


# Function for converting data types for many columns at once
def my_convert_dtype(df: pd.DataFrame, dtype_for_all: dict = None):
    """
    Ex:
    dtype_for_all = {"col_name": ["int", "int32"]}\n
    or
    dtype_for_all = {"col_name": ["int", "bool", 1, 0]}\n
    or
    dtype_for_all = {"col_name": ["bool", "int", 1, 0]}\n
    or
    dtype_for_all = {"all": ["object", "category"]}\n
    or
    dtype_for_all = {"col_name": ["object", "bool", "yes", "no"]}\n
    or
    dtype_for_all = {"col_name": ["bool", "object", "yes", "no"]}\n
    or
    dtype_for_all = {"col_name": ["object", "datetime", "%Y-%b-%d"]}\n
    dtype_for_all = {"col_name": ["object", "datetime", None]}\n


    """
    df2 = df.copy()
    if type(dtype_for_all) is dict:
        for kc, vc in dtype_for_all.items():
            if kc == "all":
                for i in df2.columns:
                    if re.match(vc[0], str(df2[i].dtypes)):
                        if vc[0] == "bool":
                            df2[i] = df2[i].map({True: vc[2], False: vc[3]})
                        elif vc[0] == "object" and vc[1] in [
                            "int",
                            "int8",
                            "int16",
                            "int32",
                            "int64",
                            "float",
                            "float8",
                            "float16",
                            "float32",
                            "float64",
                        ]:
                            print("\nPlease use my_string_handling!\n")
                        else:
                            if not vc[1] in ["bool", "datetime"]:
                                df2[i] = df2[i].astype(vc[1])
                            elif vc[1] == "datetime":
                                df2[i] = pd.to_datetime(
                                    df2[i], format=vc[2], errors="coerce"
                                )
                            elif vc[1] == "bool":
                                df2[i] = df2[i].map({vc[2]: True, vc[3]: False})
                    else:
                        df2[i] = df2[i]
            else:
                if re.match(vc[0], str(df2[kc].dtypes)):
                    if vc[0] == "bool":
                        df2[kc] = df2[kc].map({True: vc[2], False: vc[3]})
                    elif vc[0] == "object" and vc[1] in [
                        "int",
                        "int8",
                        "int16",
                        "int32",
                        "int64",
                        "float",
                        "float8",
                        "float16",
                        "float32",
                        "float64",
                    ]:
                        print("Please use my_string_handling!")
                    else:
                        if not vc[1] in ["bool", "datetime"]:
                            df2[kc] = df2[kc].astype(vc[1])
                        elif vc[1] in ["datetime"]:
                            df2[kc] = pd.to_datetime(
                                df2[kc], errors="coerce", format=vc[2]
                            )
                            if df2[kc].isnull().sum() > 0:
                                print(df2[kc][df2[kc].isnull()], "\n")
                        elif vc[1] == "bool":
                            df2[kc] = df2[kc].map({vc[2]: True, vc[3]: False})
                else:
                    df2[kc] = df2[kc]
        print("\n", df2.info(max_cols=200))
    else:
        print("Dtype has yet to be changed!")
        df2 = pd.DataFrame()
    return df2


# Function for creating ordered categorical data types
def my_create_ordinal_cat_dtype(cat_list: list, order_num_list: list):
    """DSH"""
    o = order_num_list
    ocd = pd.CategoricalDtype(
        list(pd.Series(cat_list, index=o).dropna().sort_index().values), ordered=True
    )
    return ocd


# Function for converting all columns containing ordinal categorical data into ordered categories
def my_convert_ordinal_cat_dtype(
    df: pd.DataFrame,
    col_and_order: dict,
):
    """
    Use value_counts func to create a df and sort distinct values by count. Then, use that list of distinct values for order.
    col_and_order = {
    "enrolled_university": [1, 3, 2],
            }
    """
    df1 = df.copy()
    ck = my_check_category_data(df1[list(col_and_order.keys())], count=False).replace(
        {"___": np.nan, "...": np.nan}
    )
    for i in ck.columns:
        ct_lv = my_create_ordinal_cat_dtype(list(ck[i].dropna()), col_and_order.get(i))
        df1[i] = df1[i].astype(ct_lv)
        print(
            f"'{i}': {list(df1[i].dtypes.categories)} , ordered={df1[i].dtypes.ordered}\n"
        )
    return df1


#
def my_check_dup(
    df: pd.DataFrame,
    kind="with" or "without",
    subset_col: list = "all",
    return_dup: bool = False,
):
    """Check dup"""
    df_dup = df_dup_with = df_dup_without = None
    # if not isinstance(column, list):
    #     print("Duplicate values of dataframe: ", df.duplicated(keep=False).sum(), "\n")
    if kind == "with":
        if subset_col == "all":
            print(
                "\nDuplicate values of dataframe: ",
                df.duplicated(keep=False).sum(),
                "\n",
            )
        elif isinstance(subset_col, list):
            print("\nDuplicate values of dataframe: ", df.duplicated(keep=False).sum())
            print(
                f"Duplicate values of {subset_col}:",
                df.duplicated(subset=subset_col, keep=False).sum(),
                "\n",
            )
            df_dup_with = df[df.duplicated(subset=subset_col, keep=False)]
        else:
            print(f"'column=' must be 'all' or list of dataframe's columns")
    elif kind == "without":
        if isinstance(subset_col, list):
            print("\nDuplicate values of dataframe: ", df.duplicated(keep=False).sum())
            print(
                f"Duplicate values of dataframe without {subset_col}:",
                df.duplicated(
                    subset=df.drop(columns=subset_col, inplace=False), keep=False
                ).sum(),
                "\n",
            )
            df_dup_without = df[
                df.duplicated(
                    subset=df.drop(columns=subset_col, inplace=False), keep=False
                )
            ]
        else:
            print(
                f"\nCan't find duplicates of dataframe without '{subset_col}' because 'column=' must be list of dataframe's columns.\n"
            )
    else:
        print("\nDuplicate values of dataframe: ", df.duplicated(keep=False).sum())
        print(f"\n'kind=' must be 'with' or 'without'")

    df_dup = df[df.duplicated(keep=False)]
    df_dup_w = df_dup_with if df_dup_with is not None else df_dup_without

    if return_dup is True:
        return df_dup, df_dup_w


def my_drop_dup_and_chk(
    df: pd.DataFrame,
    kind=None or "without" or "with",
    subset_col=None or list,
    keep_occurrence=False or "first" or "last",
):
    """
    keep_occurrence=False or 'first' or 'last'
    """
    df_2 = None

    if kind is None and subset_col is None:
        df_2 = df.drop_duplicates(
            keep=keep_occurrence, inplace=False, ignore_index=True
        )
    elif kind == "with" and isinstance(subset_col, list):
        df_2 = df.drop_duplicates(
            subset=subset_col, keep=keep_occurrence, inplace=False, ignore_index=True
        )
    elif kind == "without" and isinstance(subset_col, list):
        df_2 = df.drop_duplicates(
            subset=df.drop(columns=subset_col, inplace=False),
            keep=keep_occurrence,
            inplace=False,
            ignore_index=True,
        )
    else:
        print("Can't drop duplicates!'")
    if df_2 is not None:
        print("\nDrop duplicates done!")
        print(f"The dataframe contains {len(df_2)} rows after removing duplicates.\n")
        my_check_dup(df=df_2, kind=kind, subset_col=subset_col, return_dup=False)

    return df_2


def my_handle_dup_and_chk(df: pd.DataFrame, dup_column: list, handling_dict: dict):
    """
    Ex:
        handling_dict={
                    'col1': 'mean',}
    """
    df_2 = None
    df_2 = df.groupby(by=dup_column, dropna=False).agg(handling_dict).reset_index()
    print("Handling duplicates is done!")
    i = list(df.columns[~df.columns.isin(dup_column)])
    my_check_dup(df=df_2, kind="without", column=i)
    print(f"The dataframe contains {len(df_2)} rows after handling duplicates.")
    return df_2


def my_check_missing(df):
    """dd"""
    df1 = df.copy()
    df2 = df1.mask(df1 == "", inplace=False)
    miss1 = df2.isnull().sum()
    miss2 = pd.DataFrame(miss1).rename({0: "Missing"}, axis=1)
    if (miss1 == 0).all():
        return print("The data has no missing value.")
    else:
        return miss2.query("Missing > 0")


def my_handling_missing_and_chk(
    df: pd.DataFrame,
    dropna_list: list = None or "all",
    fillna_dict: dict = None,
    reindex=False,
):
    """
    Ex:\n
        fillna_dict={
                    'col1': df['col1'].mean(),
                    'col2': 'xxx'
                }\n
        dropna_list=['col3','col4']
    """
    df2 = None
    if isinstance(dropna_list, list):
        df2 = (
            df.dropna(subset=dropna_list, inplace=False)
            if fillna_dict is None
            else print("Please choose 'dropna' or 'fillna'!")
        )
    elif dropna_list is None:
        df2 = (
            df.fillna(fillna_dict, inplace=False)
            if isinstance(fillna_dict, dict)
            else print("Please choose 'dropna' or 'fillna'!")
        )
    elif dropna_list == "all":
        df2 = (
            df.dropna(inplace=False)
            if fillna_dict is None
            else print("Please choose 'dropna' or 'fillna'!")
        )
    df2 = df2.reset_index(drop=True) if reindex is True else df2
    print(my_check_missing(df2))
    return df2


# Detecting outliers
def my_iqr_detecting_outliers(df, features):
    """
    Takes a dataframe and returns an index list corresponding to the observations
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []

    for column in features:
        # 1st quartile (25%)
        # Q1 = np.percentile(df[column], 25)
        Q1 = df[column].quantile(0.25, interpolation="linear")
        # 3rd quartile (75%)
        Q3 = df[column].quantile(0.75, interpolation="linear")
        # Q3 = np.percentile(df[column], 75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determining a list of indices of outliers
        outlier_list_column = df[
            (df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)
        ].index

        # appending the list of outliers
        outlier_list.extend(y for y in outlier_list_column if y not in outlier_list)

    print("\nDetermining a list of indices of outliers: Done!\n")
    # print(
    #     column,
    #     len(outlier_list_column),
    #     "lower",
    #     (Q1 - outlier_step),
    #     "upper",
    #     (Q3 + outlier_step),
    # )

    return outlier_list


# Drop row and check
def my_drop_rows_by_index(df, index_list):
    df_2 = df.drop(index_list).reset_index(drop=True)
    print("\nDrop rows done!\n")
    return df_2


def check_common_elements_between_list(
    list1: list,
    list2: list,
    number_to_check: list,
    op="at_least" or "less_than" or "in",
):
    """
    This function checks if list1 contains at least or less than 'number_to_check' out of items from list2.

    Args:
        list1: The first list to check.
        list2: The second list to compare with.

    Returns:
        True if list1 contains at least or less than 'number_to_check' out of items from list2, False otherwise.
    """
    # Convert both lists to sets for efficient lookups
    if isinstance(list1, list) and isinstance(list2, list):
        set1 = set(list1)
        set2 = set(list2)
        # Count the number of common elements
        common_elements = len(set1.intersection(set2))
        if op == "at_least":
            return True if common_elements >= number_to_check[0] else False
        elif op == "less_than":
            return True if common_elements < number_to_check[0] else False
        elif op == "in":
            return True if common_elements in number_to_check else False
    else:
        return False


#
def my_filter_df_by_one_col(
    df: pd.DataFrame,
    col_to_filter: str,
    val1: list,
    op1: str = ">=" or dict,
    kind=None or "and" or "or",
    op2=None,
    val2: list = None,
    resetindex: bool = False,
):
    """
    op1: >= (
        or comparison operators or 'contains' or {'at_least': [1]} or {'less_than': [1]} or {'in': [1,2]} \n
        )\n
    op2 = None (or comparison operators)\n
    kind = None (or "and" or "or")\n
    val1= None (or List of ONE/MULTI value or List of regular expression (for 'contains' filtering))\n
    val2 = None (or List of ONE value)\n
    """
    if all(v is not None for v in [kind, op2, val2]):
        filtered_df = df.query(
            f"`{col_to_filter}` {op1} @val1[0] {kind} `{col_to_filter}` {op2} @val2[0]"
        )
    else:
        if op1 == "==":
            filtered_df = df[df[col_to_filter].isin(val1)]
        elif op1 in ["!=", "<>"]:
            filtered_df = df[~df[col_to_filter].isin(val1)]
        elif op1 in [">", ">=", "<", "<="]:
            filtered_df = df.query(f"`{col_to_filter}` {op1} @val1[0]")
        elif op1 == "contains":
            des_list = (
                df[df[col_to_filter].str.contains(s, case=True, na=False, regex=True)]
                for s in val1
            )
            filtered_df = pd.concat(des_list, axis=0, ignore_index=False)
        elif isinstance(op1, dict):
            for k, v in op1.items():
                k, v
            filtered_df = df[
                df[col_to_filter].apply(
                    lambda x: check_common_elements_between_list(
                        list1=x, list2=val1, op=k, number_to_check=v
                    )
                )
            ]

    return filtered_df.reset_index(drop=True) if resetindex is True else filtered_df


def my_filter_df_by_multi_col(
    df: pd.DataFrame, dict_of_params: dict, kind="and" or "or"
) -> pd.DataFrame:
    """
    #Filtering a DataFrame based on col1 and col2 and col3 and....\n
    #Filtering a DataFrame based on col1 or col2 or col3 or....\n

    dict_of_params={"col_name_1": [val1: list, op1, kind, op2, val2, resetindex],
            ....
            }\n

    op1: >= (or comparison operators)\n
    op2 = None (or comparison operators)\n
    kind = None (or "and" or "or")\n
    val1= None (or List of MULTI value)\n
    val2 = None (or List of ONE value)\n

    Examples:
        {
            "fund_c": [[40000], "<", "and", ">=", [1039], False],\n
            "acct_cur": [["euro", "vnd"], "==", None, None, None, False],
            }\n
        {
            "carriertrackingnumber": [['403C','4D57'], "contains", None, None, None, False],
            }
    """
    if kind == "and":
        df_filtered = None
        for i in dict_of_params.keys():
            df1 = df.copy() if df_filtered is None else df_filtered
            df_filtered = my_filter_df_by_one_col(
                df=df1,
                col_to_filter=i,
                val1=dict_of_params.get(i)[0],
                op1=dict_of_params.get(i)[1],
                kind=dict_of_params.get(i)[2],
                op2=dict_of_params.get(i)[3],
                val2=dict_of_params.get(i)[4],
                resetindex=dict_of_params.get(i)[5],
            )
    elif kind == "or":
        df_filtered_list = (
            my_filter_df_by_one_col(
                df=df.copy(),
                col_to_filter=i,
                val1=dict_of_params.get(i)[0],
                op1=dict_of_params.get(i)[1],
                kind=dict_of_params.get(i)[2],
                op2=dict_of_params.get(i)[3],
                val2=dict_of_params.get(i)[4],
                resetindex=dict_of_params.get(i)[5],
            )
            for i in dict_of_params.keys()
        )
        df_filtered = pd.concat(df_filtered_list, axis=0, ignore_index=False)

    return df_filtered


def my_filter_and_update(
    df: pd.DataFrame,
    filter_dict: None | dict,
    filter_kind="and" or "or",
    new_filtered_df: bool = True,
    update_dict: dict = None,
    drop_rows=False,
    index_list: list = None,
) -> pd.DataFrame:
    """
    This function enables filtering a dataframe based on one or multiple conditions in its columns or by index list. It also allows updating old values in one or more columns with new values or creating new columns (specified in the update_dict parameter) with new values or drop rows.\n
    filter_dict:
    {
        "col_name_1": [val1: list of multi, op1, kind, op2, val2: list of one, resetindex],\n
        "col_name_2": [val1: list of multi, op1, kind, op2, val2: list of one, resetindex],\n
        ....
        }\n
    update_dict:
    {
        "col_name_2": [str/num/function to update, None or int, None or dtype],\n
        "col_name_3": [str/num/function to update, position if "col_name_3" is new col, dtype if "col_name_3" is new col],\n
        ...,
        }
    """
    df2 = df.copy()
    if index_list is None:
        if filter_dict is not None:
            df_filtered = my_filter_df_by_multi_col(df2, filter_dict, kind=filter_kind)
            f = df_filtered.index
            df3 = df_filtered.copy() if new_filtered_df is True else df2.copy()
        else:
            f = df2.index
            df3 = df2.copy().loc[f] if new_filtered_df is True else df2.copy()
    else:
        f = list(index_list)
        df3 = df2.copy().loc[f] if new_filtered_df is True else df2.copy()

    # Update val or drop rows after filtering
    if update_dict is not None:
        for e in update_dict.keys():
            if e not in df3.columns:
                # Create new col with new val
                pos = (
                    update_dict.get(e)[1]
                    if update_dict.get(e)[1] is not None
                    else len(df3.columns)
                )
                df3.insert(pos, e, 0, allow_duplicates=True)
                df3[e] = (
                    df3[e].astype(update_dict.get(e)[2])
                    if update_dict.get(e)[2] is not None
                    else df3[e]
                )

            df3[e] = df3[e].mask(df3.index.isin(f), list(update_dict.get(e))[0])
            df3[e] = df3[e].convert_dtypes()

    elif drop_rows is True:
        df3 = df2.copy().drop(index=f)

    return df3


def my_multi_filter_and_update(
    df: pd.DataFrame, multi_filter_dict: dict, multi_update_dict: dict
) -> pd.DataFrame:
    """
    multi_filter_dict =
    {
    1: [{"col_name_1": [val1: list, op1, kind, op2, val2, resetindex]}, filter_kind, new_filtered_df, index_list],\n
    2: [{"col_name_1": [val1: list, op1, kind, op2, val2, resetindex]}, filter_kind, new_filtered_df, index_list]\n
    ...
    }\n
    multi_update_dict =
    {
        1: [{"col_name_1": [str/num/function to update, position if "col_name_3" is new col, None or dtype],}, drop_rows],\n
        2: [{"col_name_2": [str/num/function to update, None or int, None or dtype],}, drop_rows],
    }\n
    Ex:

    multi_filter_dict =
    {
    1: [{"channel": [["SMS"], "==", None, None, None, False]}, "and", False, None],\n
    2: [{"channel": [["Email"], "==", None, None, None, False]}, "and", False, None]
    }\n
    multi_update_dict =
    {
        1: [{"cost": [0.05, None, None],}, False],\n
        2: [{"cost2": [0.05, 1, 'int'],}, False],
    }
    """
    df_filtered = None
    for k, v in multi_filter_dict.items():
        df1 = df.copy() if df_filtered is None else df_filtered
        df_filtered = my_filter_and_update(
            df=df1,
            filter_dict=v[0],
            filter_kind=v[1],
            new_filtered_df=v[2],
            update_dict=multi_update_dict.get(k)[0],
            drop_rows=multi_update_dict.get(k)[1],
            index_list=v[3],
        )

    return df_filtered


def my_sql_join(
    ldf: pd.DataFrame,
    par: dict,
    where: dict = None,
    select: list = None,
    sort_by_dict: dict = None,
    where_kind="and" or "or",
) -> pd.DataFrame:
    """
    par: {
        1:[right_df, how, left_on/left_index, right_on/right_index, anti join]

        2: ....,

        ....
    }\n
    par = {
    1: [airbnb_price3, "left", ["listing_id"], ['listing_id'], ["left_only"]], # SQl anti left join
    2: [airbnb_last_review2, "left", ["listing_id"], ["listing_id"], None],
    }\n
    where = {
            "price_median": [[95], ">=", None, None, None, False],

            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]
            }# Multiindex column

    sort_by_dict = {
            ("price",'median'): False, # Ascending=False
        }
    Ex:
        par = {
            1: [df_2, "left", ["df_1_col1", "df_1_col2"], ["df_2_col1", "df_2_col2"], ["left_only"]
            ],# SQl anti left join

            2: [df_3, "right", ["merged_df_col1"], ["df_3_col2"], ["right_only"]
            ],# SQl anti right join

            3: [df_4, "right", ["merged_df2_col1"], ["df_4_col1"], ["right_only","left_only"]
            ]
        }

        par2 = {
        1: [r_df, "inner", ["salesorderid"], ["salesorderid"], None ],
                }

        hfc.my_sql_join(l_df, par2)

    """
    merged_df = None
    for k, v in par.items():
        if merged_df is None:
            left_df = ldf.copy()
        else:
            left_df = merged_df
        # Join by index
        if type(v[2]) is bool and type(v[3]) is bool:
            merged_df = pd.merge(
                left_df,
                v[0],
                how=v[1],
                left_index=v[2],
                right_index=v[3],
                suffixes=("_l", "_r"),
                indicator="i" + f"{k}",
            )
        # Join by cols
        elif type(v[2]) is list and type(v[3]) is list:
            merged_df = pd.merge(
                left_df,
                v[0],
                how=v[1],
                left_on=v[2],
                right_on=v[3],
                suffixes=("_l", "_r"),
                indicator="i" + f"{k}",
            )
        else:
            merged_df = ldf.copy()

        # Filter indicator col:
        # Example: "left_only" -> SQL left anti join
        merged_df = (
            my_filter_df_by_one_col(
                merged_df,
                "i" + f"{k}",
                v[4],
                op1="==",
                kind=None,
                op2=None,
                val2=None,
                resetindex=False,
            )
            if v[4] is not None
            else merged_df
        )

    # where - select
    merged_df = (
        my_filter_df_by_multi_col(merged_df, where, kind=where_kind)
        if where is not None
        else merged_df
    )

    merged_df = (
        merged_df.sort_values(
            by=list(sort_by_dict.keys()),
            ascending=list(sort_by_dict.values()),
            inplace=False,
        )
        if sort_by_dict is not None
        else merged_df
    )
    merged_df = merged_df[select] if select is not None else merged_df

    return merged_df


def my_sql_groupby_agg_having(
    df: pd.DataFrame,
    where_dict: dict = None,
    where_kind="and" or "or",
    col_to_group=list,
    index_to_group: int | str | list = None,
    group_dropna=False,
    agg_dict: dict = None,
    unstack: list = None,
    multi_level_col=False,
    reset_idx_lv: list = None,
    multi_level_index=True,
    having=None or dict or list,
    having_kind="and" or "or",
    total_row: bool = False,
    total_col: bool = False,
    normalize=None or "col" or "row",
    sort_vals: dict = None,
    sort_idx: dict = None,
) -> pd.DataFrame:
    """
    Function for grouping by cols or different levels of a hierarchical index,...
    Ex:
        agg_dict = {
            "price": ["mean", "median", "count"], # Use standard functions\n
            "price": custom_func, # Or use custom functions\n
            "price_per_month": ["sum"]\n
            }\n



        where_dict = {
            "price_median": [[95], ">=", None, None, None, False],\n
            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]\n
            }# Multiindex column\n
            "fund_c": [[40000], "<", "and", ">=", [1039], False],\n

        having = {
            "price_median": [[95], ">=", None, None, None, False],\n
            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]\n
            }# Multiindex column\n
        having = [index_list] # filter df by list of index\n
        sort_idx={1: True}\n
        sort_vals = {
            ("price",'median'): False, # Ascending=False\n
        }\n

        unstack=[0,1] # Unstack after group, level 0 and 1\n
        multi_level_col = True | False (join col name) | dict (rename col by dict) \n
        multi_level_index = True | False | list (reset index by list of levels)\n
    """
    # Filter df (select-from-where in SQL)
    df0 = (
        df.copy()
        if where_dict is None
        else my_filter_df_by_multi_col(df.copy(), where_dict, kind=where_kind)
    )

    # Group-Agg
    df1 = (
        df0.groupby(by=col_to_group, dropna=group_dropna, observed=False).agg(agg_dict)
        if index_to_group is None
        else df0.groupby(level=index_to_group, dropna=group_dropna, observed=False).agg(
            agg_dict
        )
    )

    df1 = df1.unstack(level=unstack) if unstack is not None else df1

    # Drop last level from multi-level columns and join others by '_' or rename col
    if multi_level_col is False:
        df1.columns = ["_".join(col) for col in df1.columns]
    elif isinstance(multi_level_col, dict):
        df1.columns = ["_".join(col) for col in df1.columns]
        df1 = df1.rename(columns=multi_level_col)

    # reset index level
    if isinstance(reset_idx_lv, list):
        df1.reset_index(level=reset_idx_lv, inplace=True)

    # flatten row index
    if multi_level_index is False:
        df1.index = [re.compile(r"[\(\)\']+").sub("", str(i)) for i in df1.index.values]

    # Filter df after grouping (having in SQL)
    if having is not None:
        df1 = (
            my_filter_df_by_multi_col(df1, having, kind=having_kind)
            if isinstance(having, dict)
            else df1.loc[having]
        )

    if total_row is True:
        t1 = [" " for i in df1.index.names]
        t1.insert(0, "total")
        t1.pop()
        t1 = tuple(t1) if len(t1) > 1 else "total"
        df1.loc[t1, :] = df1.sum()
        if normalize == "row":
            for i in df1.index:
                df1.loc[i, :] = (
                    (df1.loc[i, :] * 100 / df1.loc[t1, :])
                    .apply(lambda x: round(x, 1))
                    .fillna(0)
                )
            df1 = df1.rename(index={"total": "total(%)"}, level=0)

    if total_col is True:
        t2 = [" " for i in df1.columns.names]
        t2.insert(0, "total")
        t2.pop()
        t2 = tuple(t2) if len(t2) > 1 else "total"
        df1[t2] = df1.sum(axis=1)
        if normalize == "col":
            for i in df1.columns:
                df1[i] = (df1[i] * 100 / df1[t2]).apply(lambda x: round(x, 1)).fillna(0)
            df1 = df1.rename(columns={"total": "total(%)"}, level=0)

    df1 = (
        df1.sort_values(
            by=list(sort_vals.keys()),
            ascending=list(sort_vals.values()),
            inplace=False,
        )
        if sort_vals is not None
        else df1
    )

    df1 = (
        df1.sort_index(
            level=list(sort_idx.keys()),
            ascending=list(sort_idx.values()),
            inplace=False,
        )
        if sort_idx is not None
        else df1
    )

    # remove columns index name
    df1.columns.names = [None for n in list(df1.columns.names)]

    return df1


def my_sql_groupby_having_win_func(
    df: pd.DataFrame,
    where_dict: dict = None,
    where_kind="and" or "or",
    col_to_group=list,
    index_to_group: int | str | list = None,
    group_dropna=False,
    col_to_trans_and_func: dict = None,
    having=None or dict or list,
    having_kind="and" or "or",
    sort_vals: dict = None,
    sort_idx: dict = None,
    total_row: bool = False,
) -> pd.DataFrame:
    """
    Ex:
        col_to_trans_and_name = {
            "price": "avg_price",
            ...
            }

        col_to_trans_and_func =
            {
            "price":["mean", "new_col_name1"],\n
            "cost": [lambda x: x.max() - x.min(), "new_col_name1"],
                    }
        where_dict = {
            "price_median": [[95], ">=", None, None, None, False],\n
            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]\n
            }# Multiindex column\n
        having = {
            "price_median": [[95], ">=", None, None, None, False],\n
            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]\n
            }# Multiindex column\n
        having = [index_list] # filter df by list of index\n
        sort_by_dict = {
            ("price",'median'): False, # Ascending=False\n
        }\n
    """
    df1 = df2 = (
        df.copy()
        if where_dict is None
        else my_filter_df_by_multi_col(df.copy(), where_dict, kind=where_kind)
    )

    for k, v in col_to_trans_and_func.items():
        df1[v[1]] = (
            df1.groupby(col_to_group, dropna=group_dropna, observed=False)[
                [k]
            ].transform(str(v[0]))
            if index_to_group is None
            else df1.groupby(level=index_to_group, dropna=group_dropna, observed=False)[
                [k]
            ].transform(str(v[0]))
        )
        break

    df3 = pd.concat(
        [
            df2.groupby(
                by=col_to_group,
                level=index_to_group,
                dropna=group_dropna,
                observed=False,
            )[[k]].transform(str(v[0]))
            for k in col_to_trans_and_func.keys()
        ],
        axis=1,
        ignore_index=False,
    ).rename(columns={k: v[1]})

    df4 = pd.concat(
        [df1, df3],
        axis=1,
        ignore_index=False,
    )

    # Drop Duplicate Columns
    df1 = df4.loc[:, ~df4.columns.duplicated()]
    # Filter df after groupby
    if having is not None:
        df1 = (
            my_filter_df_by_multi_col(df1, having, kind=having_kind)
            if isinstance(having, dict)
            else df1.loc[having]
        )
    # Sort value and/or index
    df1 = (
        df1.sort_values(
            by=list(sort_vals.keys()),
            ascending=list(sort_vals.values()),
            inplace=False,
        )
        if sort_vals is not None
        else df1
    )
    df1 = (
        df1.sort_index(
            level=list(sort_idx.keys()),
            ascending=list(sort_idx.values()),
            inplace=False,
        )
        if sort_idx is not None
        else df1
    )

    if total_row is True:
        t1 = [" " for i in df1.index.names]
        t1.insert(0, "total")
        t1.pop()
        t1 = tuple(t1) if len(t1) > 1 else "total"
        df1.loc[t1, :] = df1.sum()

    return df1


def my_pivot_agg_filter(
    df: pd.DataFrame,
    # col_to_agg: list or str,
    col_to_index: list | str,
    col_to_col: list | str,
    agg_dict: dict | str,
    fillna=None,
    # margins=["All", False],
    group_dropna=True,
    having_dict: dict = None,
    sort_vals: dict = None,
    sort_idx: dict = None,
    multi_level_col=False,
    having_kind="and" or "or",
) -> pd.DataFrame:
    """
    Function for grouping to pivot table by cols or different levels of a hierarchical index,...
    Ex:
        agg_dict = {
            "price": ["mean", "median", "count"], # Use standard functions\n
            "price": custom_func, # Or use custom functions\n
            "price_per_month": ["sum"]\n
            }\n
        having_dict = {
            "price_median": [[95], ">=", None, None, None, False],\n
            ("price_per_month", "sum"): [[386.25], "==", None, None, None, False]\n
            }# Multiindex column\n
        sort_vals = {
            ("price",'median'): False, # Ascending=False\n
        }\n
        sort_idx = {
            "color": False, # Ascending=False\n
            "breed": True
        }\n
    """
    df1 = df.copy()
    df1 = df1.pivot_table(
        # values=col_to_agg,
        index=col_to_index,
        columns=col_to_col,
        aggfunc=agg_dict,
        fill_value=fillna,
        # margins=margins[1],
        # margins_name=margins[0],
        dropna=group_dropna,
        observed=False,
    )

    # Drop First Level From Multi Level Columns and add level label as prefix
    df1.columns = (
        ["_".join(col) for col in df1.columns]
        if multi_level_col is False
        else df1.columns
    )

    df1 = (
        my_filter_df_by_multi_col(df1, having_dict, kind=having_kind)
        if having_dict is not None
        else df1
    )
    df1 = (
        df1.sort_values(
            by=list(sort_vals.keys()),
            ascending=list(sort_vals.values()),
            inplace=False,
        )
        if sort_vals is not None
        else df1
    )
    df1 = (
        df1.sort_index(
            level=list(sort_idx.keys()),
            ascending=list(sort_idx.values()),
            inplace=False,
        )
        if sort_idx is not None
        else df1
    )
    return df1


#
def my_drop_col_and_chk(
    df: pd.DataFrame,
    drop_cols: list,
    chkkind=None or "with" or "without",
    chkcolumn=None or "all" or list,
) -> pd.DataFrame:
    """Drop column and check duplicates"""
    df2 = df.drop(columns=drop_cols, inplace=False)
    print("Drop columns done!")
    (
        None
        if chkkind is None or chkcolumn is None
        else my_check_dup(df=df2, kind=chkkind, subset_col=chkcolumn)
    )
    # print(df_2)
    return df2


#
def my_binning_data(
    df: pd.DataFrame,
    cut_and_name: dict,
    qcut: int | list = None,
    bins=list,
    label_names=None or list or False,
    labels_ordered: bool = True,
    right=False,
    precision: int = 3,
) -> pd.DataFrame:
    """
    Ex: Budget -> 1-69; Average -> 70-175, Expensive -> 176-350, Extravagant -> >350\n
        # cut:\n
        cut_and_name = {"col_to_cut":"new_cut_col"}
        bins = [0, 69, 175, 350, np.inf]\n
        label_names = ["Budget", "Average", "Expensive", "Extravagant"]\n
        right = True\n
        # qcut:\n
            terciles: qcut=[0, 1/3, 2/3, 1] or qcut=3
            quintiles: qcut=[0, .2, .4, .6, .8, 1] or qcut=5
            sextiles: qcut=[0, 1/6, 1/3, .5, 2/3, 5/6, 1] or qcut=6
    """
    df1 = df.copy()
    for k, v in cut_and_name.items():
        df1[v] = (
            pd.cut(
                df1[k].array,
                bins=bins,
                labels=label_names,
                right=right,
                # retbins=True,
                precision=precision,
                ordered=labels_ordered,
            )
            if qcut is None
            else pd.qcut(
                df1[k],
                q=qcut,
                labels=label_names,
                right=right,
                # retbins=True,
                precision=precision,
            )
        )
        df1[v].dtypes
    return df1


def my_fuzzy_string_replace(df: pd.DataFrame, col: str, params_dict: dict):
    """
    col: column to check\n
    params_dict:\n
    {\n"correct_str1": [fuzz.WRatio, similarity score],\n "correct_str2": [fuzz.WRatio, similarity score],\n...}
    """
    # For each correct category
    new_col = col + "_new"
    df2 = df.copy()
    df2[new_col] = df2[col]
    df2.insert(len(df2.columns), "similarity_score", 0, allow_duplicates=True)
    for k, v in params_dict.items():
        # Find potential matches in states with typoes
        matches = process.extract(k, df2[col], scorer=v[0], limit=df2.shape[0])
        # For each potential match match
        for potential_match in matches:
            # If high similarity score
            if potential_match[1] >= v[1]:
                # Replace typo with correct category
                df2.loc[df2[col] == potential_match[0], new_col] = k
                df2.loc[df2[col] == potential_match[0], "similarity_score"] = (
                    potential_match[1]
                )
    return df2


def my_string_handling(
    df: pd.DataFrame,
    regex_to_replace: dict = None,
    to_split: dict = None,
    to_join: dict = None,
    changecase=None or dict,
    strip_char=None or dict,
    strip_accents=False,
) -> pd.Series | list:
    """
    Ex1:\n
        email_regex = r'[\w\.-]+@[\w\.-]+' (# Works with all email addresses with standard English characters)\n
        phone_regex = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]' (# Works for most international phone numbers)

    Ex2:\n
        regex_to_replace =
            { "col_name1":
                ["new_col1", {   r"[^a-zA-Z0-9 _&-+]+":"",
                    r"[ &-+]+"  :   "_",
                    "ab" : "CD"}],
              "col_name2":
                ["new_col1", {email_regex   :  " ",
                    "xxx": "XXXX"}],
            }\n

        to_split=
        {"date_string_col": [r"/", True, {2: "year"}],} \n
        #split string of "date_string_col" by pat "/", and expand (True), then get the 3rd substring to new col "year"\n
        to_split=
        {"date_string_col": [r"/", False, None],} \n
        #split string of "date_string_col" by pat "/", return date_string_col containing lists of strings\n
        to_split={"year": [0, 2, "ye"]}\n
        # split string of "year" col by index of char (from 0 to 2) to new col "ye"\n

        to_join=
        {"new_string_col1": ["-", ["col1", "col2",...]],
        ....
        } \n

        strip_char={
            "col1": [" "],
            "col1": [":"],
            ...}\n
            ={
                "all": [" ", ":"],
            }\n

        changecase={
            "col1": "lower" / "upper" / "title" / "capitalize" / "swapcase",
            ...}\n
            ={
                "all": "lower" / "upper" / "title" / "capitalize" / "swapcase"
            }\n

    """
    df2 = df.copy()

    if strip_accents is True:
        for ser, vr in regex_to_replace.items():
            df2[vr[0]] = (
                my_strip_ascents(
                    df2[ser].astype(str),
                )
                if df2[ser].dtype == "object"
                else df2[ser]
            )

    if regex_to_replace is not None:
        for ser, vr in regex_to_replace.items():
            s1 = None
            df2[vr[0]] = df2[ser].astype(str)
            for k, v in vr[1].items():
                y = re.compile(rf"{k}")
                s1 = (
                    df2[vr[0]].replace(y, v, regex=True).astype(type(v))
                    if s1 is None
                    else s1.replace(y, v, regex=True).astype(type(v))
                )
            df2[vr[0]] = s1

    if to_split is not None:
        # s12 = None
        for ser2, vr2 in to_split.items():
            df2[ser2] = df2[ser2].astype(str)
            if not isinstance(vr2[0], int):
                y2 = re.compile(rf"{vr2[0]}")
                s12 = df2[ser2].str.split(y2, expand=vr2[1], regex=True)
                if isinstance(vr2[2], dict) and vr2[1] is True:
                    s121 = s12.rename(columns=vr2[2])
                    s121 = s121[list(vr2[2].values())]
                else:
                    s121 = s12
            else:
                s12 = df2[ser2].str[vr2[0] : vr2[1]]
                s121 = (
                    pd.DataFrame(s12).rename(columns={ser2: vr2[2]})
                    if isinstance(vr2[2], str)
                    else s12
                )

            if isinstance(s121, pd.DataFrame):  # ~expand=True --> ouput: DataFrame
                df2 = pd.concat([df2, s121], axis=1)
            else:
                df2[f"{ser2}_s"] = s121  # ~expand=False --> ouput: Series

    #  to_join=
    #         {"new_string_col1": ["-", ["col1", "col2",...]],
    #         ....
    #         } \n

    if to_join is not None:
        # s13 = None
        for ser3, vr3 in to_join.items():
            df2[ser3] = df2[vr3[1]].apply(vr3[0].join, axis=1)

    if strip_char is not None:
        for ser, vr in strip_char.items():
            if ser != "all":
                for t2 in vr:
                    df2[ser] = df2[ser].str.strip(t2).str.strip()
            else:
                for ser in df2.columns:
                    for t2 in vr:
                        df2[ser] = df2[ser].str.strip(t2).str.strip()

    if changecase is not None:
        for ser, vr in changecase.items():
            if ser != "all":
                if vr == "lower":
                    df2[ser] = df2[ser].str.lower()
                elif vr == "upper":
                    df2[ser] = df2[ser].str.upper()
                elif vr == "title":
                    df2[ser] = df2[ser].str.title()
                elif vr == "capitalize":
                    df2[ser] = df2[ser].str.capitalize()
                elif vr == "swapcase":
                    df2[ser] = df2[ser].str.swapcase()
            else:
                for ser2 in df2.columns:
                    df2[ser2] = df2[ser2].astype(str)
                    if vr == "lower":
                        df2[ser2] = df2[ser2].str.lower()
                    elif vr == "upper":
                        df2[ser2] = df2[ser2].str.upper()
                    elif vr == "title":
                        df2[ser2] = df2[ser2].str.title()
                    elif vr == "capitalize":
                        df2[ser2] = df2[ser2].str.capitalize()
                    elif vr == "swapcase":
                        df2[ser2] = df2[ser2].str.swapcase()

    print(
        my_check_category_data(
            df2,
            count=False,
            normalize=False,
            num_to_cat_cutoff=0,
        )
    )
    return df2


def my_strip_ascents(txt: str | pd.Series) -> str:
    char_with_accents = "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÄÅÇĐÈÉẺẼẸÊẾỀỂỄỆËÍÌỈĨỊÎÏÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÖÚÙỦŨỤƯỨỪỬỮỰÛÜÝỲỶỸỴÑáàảãạăắằẳẵặâấầẩẫậäåçđèéẻẽẹêếềểễệëíìỉĩịîïóòỏõọôốồổỗộơớờởỡợöúùủũụưứừửữựûüýỳỷỹỵñ"
    alphabet_char = (
        "A" * 19
        + "C"
        + "D"
        + "E" * 12
        + "I" * 7
        + "O" * 18
        + "U" * 13
        + "Y" * 5
        + "N"
        + "a" * 19
        + "c"
        + "d"
        + "e" * 12
        + "i" * 7
        + "o" * 18
        + "u" * 13
        + "y" * 5
        + "n"
    )
    # 8 kí tự dấu dưới dạng unicode chuẩn D
    d_unicode_ascents_char = (
        chr(774)
        + chr(770)
        + chr(795)
        + chr(769)
        + chr(768)
        + chr(777)
        + chr(771)
        + chr(803)
    )
    strip_table = str.maketrans(
        char_with_accents, alphabet_char, d_unicode_ascents_char
    )
    txt2 = (
        txt.translate(strip_table)
        if isinstance(txt, str)
        else txt.str.translate(strip_table)
    )
    return txt2


############### Statistical test ###############
# Chi-squared test
def my_chi_square_test(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    correction: bool = True,
    threshold: float = 0.05,
    hypothesis_testing: bool = False,
):
    """
    This test can be used to:
    - evaluate the quality of a categorical variable in a classification problem or to
    - check the similarity between two categorical variables.\n
    For example:
    - a good categorical predictor and the class column should present high chi^2 and low p-value.
    - similar categorical variables should present low chi^2 and high p-value.
    """
    expected, observed, stats = pg.chi2_independence(
        data=df.copy(), y=x_col, x=y_col, correction=correction
    )
    r = pd.DataFrame.from_dict(
        {
            "chi^2": [stats.chi2.max()],
            "p-value": [stats.pval.max()],
            "alpha": [threshold],
        }
    )
    if stats.pval.max() < threshold:
        print("\nChi-squared test result: The p-value is less than alpha.\n")
        (
            print("\nHypothesis testing result: The 'Null hypothesis' is rejected!\n")
            if hypothesis_testing is True
            else None
        )
    else:
        print("\nChi-squared test result: The p-value is not less than alpha.\n")
        (
            print(
                "\nHypothesis testing result: The 'Null hypothesis' failed to be rejected!\n"
            )
            if hypothesis_testing is True
            else None
        )
    return r
