"""Functions for Data Preprocessing, ML Modeling and Plotting"""

import re, time, math
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
import numpy as np
import pandas as pd

# Machine Learning: Model Helpers
import sklearn.model_selection as sklms
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import sklearn.feature_selection as sklfs
import sklearn.preprocessing as sklpr

# import sklearn.compose as sklco
# import sklearn.pipeline as sklpi
import imblearn.over_sampling as imbos
from shap import TreeExplainer

# Machine Learning: Model Algorithms
import sklearn.linear_model as sklli

# import sklearn.naive_bayes as sklnb
# import sklearn.tree as skltr
# import sklearn.svm as sklsv
# import sklearn.ensemble as sklen
# import sklearn.neighbors as sklne
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Machine Learning: Score Functions & Perfomance Metrics
import sklearn.metrics as sklme

############################################################################


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


# Input - Output Separating
# @timing__decorator
def my_data_separation(
    df: pd.DataFrame, output: str, test_data_size=0.2, random_state=1981
):
    """Separate Training and Validation Datasets"""

    X, y = my_sep_feature_label(df.copy(), output, pr=False)

    X_tr, X_val, y_tr, y_val = sklms.train_test_split(
        X, y, test_size=test_data_size, random_state=random_state, stratify=y
    )

    train_dataset = pd.concat([X_tr, y_tr], axis=1)
    test_dataset = pd.concat([X_val, y_val], axis=1)

    print("\n(+) Data Separation (with stratify parameter):\n")
    print(
        "Unseen dataset: {} rows. Percentage of each class in '{}' variable: {}".format(
            len(X_val),
            output,
            dict(
                y_val.value_counts(normalize=True)
                .round(3)
                .mul(100)
                .round(0)
                .astype(str)
                + "%"
            ),
        )
    )
    print(
        "Train dataset: {} rows. Percentage of each class in '{}' variable: {}".format(
            len(X_tr),
            output,
            dict(
                y_tr.value_counts(normalize=True).round(3).mul(100).round(0).astype(str)
                + "%"
            ),
        )
    )
    return train_dataset, test_dataset


def my_sep_feature_label(df, output, pr=True):
    X = df.drop(columns=output, inplace=False)
    y = df[output]
    if pr is True:
        print("\n(+) Input - Output Separation:\n")
        print(
            "X (input) dataset: {} rows x {} columns.\n ({})".format(
                len(X), len(X.columns), list(X.columns.values)
            )
        )
        print(
            "y (output) dataset: {} rows x 1 columns.\n (['{}'])".format(len(y), output)
        )
    return X, y


# Train-Test Split
# @timing__decorator
def my_train_test_split(df, output, test_size, random_state):
    """"""

    X, y = my_sep_feature_label(df, output, pr=True)

    X_tr, X_te, y_tr, y_te = sklms.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("\n(+) Data Splitting (with stratify parameter):\n")
    print(
        "X_valid, y_valid: {} rows. Percentage of each class in y_test: {}".format(
            len(X_te),
            dict(
                y_te.value_counts(normalize=True).round(3).mul(100).round(0).astype(str)
                + "%"
            ),
        )
    )
    print(
        "X_train, y_train: {} rows. Percentage of each class in y_train: {}".format(
            len(X_tr),
            dict(
                y_tr.value_counts(normalize=True).round(3).mul(100).round(0).astype(str)
                + "%"
            ),
        )
    )
    return X_tr, X_te, y_tr, y_te


# Splitting-Balacing-Scaling
# @timing__decorator
def my_features_scaling(X_train, X_test, scaler="None"):
    """
    - Fit the scaler using available training data.
    - Apply the scale to training data.
    - Apply the scale to test data.
    """
    # if balacing is not None:  # (only for training data, and after splitting)
    #     X_train, y_train, X_test, y_test

    if scaler is not None:
        sc = scaler
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
        print("\n(+) Feature Scaling:\n")
        print("X_train scaled:\n", X_train_sc[:2, :], "\n...")
        print("X_test scaled:\n", X_test_sc[:2, :], "\n...")
        return X_train_sc, X_test_sc, sc
    else:
        return X_train, X_test


#####################################################################################
# SHAP method for Feature Selection
# https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30
# https://towardsdatascience.com/your-features-are-important-it-doesnt-mean-they-are-good-ff468ae2e3d4
#####################################################################################
def shap_sum2proba(shap_sum):
    """Compute sigmoid function of the Shap sum to get predicted probability."""
    return 1 / (1 + np.exp(-shap_sum))


def individual_log_loss(y_true, y_pred, eps=1e-15):
    """Compute log-loss for each individual of the sample."""

    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def get_preds_shaps(X, y, model):
    # def get_preds_shaps(X, y, X_valid, y_valid, model):
    """Get predictions (predicted probabilities) and SHAP values for a dataset."""
    shap_model = model.fit(X, y)
    preds = pd.Series(shap_model.predict_proba(X)[:, 1], index=X.index)
    shap_explainer = TreeExplainer(shap_model)
    shap_expected_value = shap_explainer.expected_value[-1]
    shaps = pd.DataFrame(
        data=shap_explainer.shap_values(X)[1], index=X.index, columns=X.columns
    )

    # predsv = pd.Series(shap_model.predict_proba(X_valid)[:, 1], index=X_valid.index)
    # shap_explainerv = TreeExplainer(shap_model)
    # shap_expected_valuev = shap_explainerv.expected_value[-1]
    # shapsv = pd.DataFrame(
    #     data=shap_explainerv.shap_values(X_valid)[1],
    #     index=X_valid.index,
    #     columns=X_valid.columns,
    # )
    # return preds, shaps, shap_expected_value, predsv, shapsv, shap_expected_valuev
    return preds, shaps, shap_expected_value


def get_feature_contributions(y_true, y_pred, shap_values, shap_expected_value):
    """Compute prediction contribution and error contribution for each feature."""

    prediction_contribution = shap_values.abs().mean().rename("Prediction")

    ind_log_loss = individual_log_loss(y_true=y_true, y_pred=y_pred).rename("log_loss")
    y_pred_wo_feature = shap_values.apply(
        lambda feature: shap_expected_value + shap_values.sum(axis=1) - feature
    ).map(shap_sum2proba)
    ind_log_loss_wo_feature = y_pred_wo_feature.apply(
        lambda feature: individual_log_loss(y_true=y_true, y_pred=feature)
    )
    ind_log_loss_diff = ind_log_loss_wo_feature.apply(
        lambda feature: ind_log_loss - feature
    )
    error_contribution = ind_log_loss_diff.mean().rename("Error").T

    return prediction_contribution, error_contribution


@timing__decorator
def my_shap_method(shap_X_train, shap_y_train, shap_model):
    # def my_shap_method(
    #     shap_X_train, shap_X_valid, shap_y_train, shap_y_valid, shap_model=LGBMClassifier()
    # ):
    (
        preds,
        shaps,
        shap_expected_value,
        # predsv,
        # shapsv,
        # shap_expected_valuev,
    ) = get_preds_shaps(
        X=shap_X_train,
        y=shap_y_train,
        # X_valid=shap_X_valid,
        # y_valid=shap_y_valid,
        model=shap_model,
    )

    assert (
        (preds - (shap_expected_value + shaps.sum(axis=1)).apply(shap_sum2proba)).abs()
        < 1e-10
    ).all()

    prediction_contribution_trn, error_contribution_trn = get_feature_contributions(
        y_true=shap_y_train,
        y_pred=preds,
        shap_values=shaps,
        shap_expected_value=shap_expected_value,
    )

    # prediction_contribution_val, error_contribution_val = get_feature_contributions(
    #     y_true=shap_y_valid,
    #     y_pred=predsv,
    #     shap_values=shapsv,
    #     shap_expected_value=shap_expected_valuev,
    # )

    contributions_trn = pd.concat(
        [prediction_contribution_trn, error_contribution_trn], axis=1
    )

    contributions_trn.insert(
        0,
        "Error_Rank",
        contributions_trn["Error"].rank(method="dense", ascending=True).astype("int"),
    )

    contributions_trn = contributions_trn.sort_values("Error", ascending=True)

    contributions_trn = pd.concat(
        [contributions_trn], keys=["Train_Data"], names=["Contribution"], axis=1
    )

    # contributions_val = pd.concat(
    #     [prediction_contribution_val, error_contribution_val], axis=1
    # )

    # contributions_val.insert(
    #     0,
    #     "Error_Rank",
    #     contributions_val["Error"].rank(method="dense", ascending=True).astype("int"),
    # )

    # contributions_val = pd.concat(
    #     [contributions_val], keys=["Valid_Data"], names=["Contribution"], axis=1
    # )

    # shap_results = pd.merge(
    #     contributions_trn,
    #     contributions_val,
    #     how="inner",
    #     left_index=True,
    #     right_index=True,
    # )
    shap_results = contributions_trn
    print(f"\n(+) SHAP method: Prediction Contribution and Error Contribution.\n")
    return display(shap_results)


#######################################################################################


# VIF calculation
def calculate_vif(df, features):
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        a = [f for f in features if f != feature]
        a, b = df[a], df[feature]
        # extract r-squared from the fit
        r2 = sklli.LinearRegression().fit(a, b).score(a, b)

        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1 / (tolerance[feature])
        df_vif = pd.DataFrame({"VIF": vif, "Tolerance": tolerance})
        df_vif = df_vif.round(3).sort_values(by=["VIF"], ascending=False)
    # return VIF DataFrame
    return df_vif


# KBest Mutual Info for Classification
@timing__decorator
def my_kbest_mutual_clf(X_train, X_valid, y_train):
    k = "all"
    bestfeatures = sklfs.SelectKBest(score_func=sklfs.mutual_info_classif, k=k)
    bestfeatures.fit(X_train, y_train)
    # transform train input data
    bestfeatures.transform(X_train)
    bestfeatures.transform(X_valid)
    dfscores = pd.DataFrame(bestfeatures.scores_)
    dfcolumns = pd.DataFrame(bestfeatures.get_feature_names_out())
    featureScores = pd.merge(
        dfcolumns, dfscores, how="inner", left_index=True, right_index=True
    )
    # concat two dataframes: join inner by key is index
    featureScores.columns = ["Feature", "Score"]  # naming the dataframe columns
    featureScores = featureScores.nlargest(len(dfcolumns), "Score").reset_index(
        drop=True
    )
    featureScores.set_index(["Feature"], inplace=True)
    featureScores["MF_Rank"] = (
        featureScores["Score"].rank(method="dense", ascending=False).astype("int")
    )
    return featureScores


@timing__decorator
def my_rfe_clf(X_train, y_train, estimator, n_features_to_select):
    bestfeaturesRFE = sklfs.RFE(
        # estimator=skltr.DecisionTreeClassifier(random_state=1981),
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=1,
        verbose=0,
    )
    fit = bestfeaturesRFE.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.ranking_)
    # dfcolumns = pd.DataFrame(fit.get_feature_names_out())
    dfcolumns = pd.DataFrame(X_train.columns)  # de xep hang duoc

    featureScores = pd.merge(
        dfcolumns, dfscores, how="inner", left_index=True, right_index=True
    )
    # concat two dataframes: join inner by key is index
    featureScores.columns = ["Feature", "RFE_Rank"]  # naming the dataframe columns
    featureScores = featureScores.nsmallest(
        len(dfcolumns), "RFE_Rank", keep="all"
    ).reset_index(drop=True)
    featureScores.set_index(["Feature"], inplace=True)
    return featureScores


@timing__decorator
def my_rfecv(
    X_train,
    y_train,
    estimator,
    starkf_cv=5,
    scoring=str,
    all_features=list,
    min_features_to_select=1,
):
    """Recursive feature elimination with cross-validation
    cv = StratifiedKFold(starkf_cv)

    """
    bestfeaturesRFE = sklfs.RFECV(
        # estimator=skltr.DecisionTreeClassifier(random_state=1981),
        estimator=estimator,
        min_features_to_select=min_features_to_select,
        cv=sklms.StratifiedKFold(starkf_cv, random_state=1981, shuffle=True),
        n_jobs=-1,
        scoring=scoring,
        step=1,
        verbose=0,
    )
    fit = bestfeaturesRFE.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.ranking_)
    # dfcolumns2 = pd.DataFrame(fit.get_feature_names_out())
    dfcolumns = pd.DataFrame(all_features)  # for output

    featureScores = pd.merge(
        dfcolumns, dfscores, how="inner", left_index=True, right_index=True
    )
    # concat two dataframes: join inner by key is index
    featureScores.columns = ["Features", "RFECV-Rank"]  # naming the dataframe columns
    featureScores = featureScores.nsmallest(
        len(dfcolumns), "RFECV-Rank", keep="all"
    ).reset_index(drop=True)
    featureScores.set_index(["Features"], inplace=True)

    # Plot number of features VS. cross-validation scores
    n_scores = len(fit.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        fit.cv_results_["mean_test_score"],
        yerr=fit.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

    print(
        f"\n(+) Recursive feature elimination with cross-validation:\n\nOptimal number of features: {fit.n_features_}\n"
    )
    return display(featureScores)


################################################################################
# The 'scoring' parameter of check_scoring must be a str among {'precision_micro', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error', 'v_measure_score', 'roc_auc', 'average_precision', 'precision_weighted', 'neg_negative_likelihood_ratio', 'jaccard_samples', 'recall_macro', 'neg_mean_poisson_deviance', 'recall', 'adjusted_mutual_info_score', 'fowlkes_mallows_score', 'recall_samples', 'matthews_corrcoef', 'recall_weighted', 'f1_macro', 'adjusted_rand_score', 'jaccard_weighted', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'r2', 'balanced_accuracy', 'rand_score', 'neg_log_loss', 'mutual_info_score', 'recall_micro', 'neg_mean_gamma_deviance', 'roc_auc_ovo_weighted', 'jaccard_macro', 'homogeneity_score', 'max_error', 'neg_mean_squared_error', 'precision_macro', 'top_k_accuracy', 'neg_brier_score', 'neg_root_mean_squared_log_error', 'completeness_score', 'roc_auc_ovr', 'jaccard', 'positive_likelihood_ratio', 'precision_samples', 'normalized_mutual_info_score', 'neg_mean_squared_log_error', 'f1_samples', 'f1_micro', 'jaccard_micro', 'precision', 'f1_weighted', 'accuracy', 'explained_variance', 'f1'}

################################################################################
# Functions and Plot for ML Modeling


# Learning Curve
# @timing__decorator
def my_plot_learning_curve(
    estimator,
    name,
    X,
    y,
    ax,
    cv,
    scoring=str,
    train_sizes=np.linspace(0.1, 1.0, 5),
    plot_name=str,
):
    """Plot the learning curves for an estimator in the specified axes object."""

    train_sizes, train_scores, test_scores = sklms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="dodgerblue",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="darkorange",
    )
    ax.plot(
        train_sizes,
        train_scores_mean,
        color="dodgerblue",
        marker="o",
        linestyle="-",
        label="Training Score",
    )
    ax.plot(
        train_sizes,
        test_scores_mean,
        color="darkorange",
        marker="o",
        linestyle="-",
        label="Cross-validation Score",
    )
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score (" + scoring + ")")
    ax.legend(loc="best")
    ax.set_title(
        f"Learning Curves on the {plot_name}\n({name})",
    )


# Confusion Matrix
# @timing__decorator
def my_plot_conf_mx(y_true, y_pred, model_name, df_name, ax):
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # _, ax = plt.subplots(figsize=(6, 4))
    cm = sklme.confusion_matrix(y_true, y_pred)
    _ = sns.heatmap(
        cm,
        cmap=colormap,
        square=True,
        cbar_kws={"shrink": 0.9},
        ax=ax,
        fmt="g",
        annot=True,
        linewidths=0.1,
        # vmax=1.0,
        linecolor="white",
        annot_kws={"fontsize": 10},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix for the " + df_name + " \n(" + model_name + ")", y=1)


# ROC Curve
# @timing__decorator
def my_plot_roc(X, y_true, pos_label, model_fitted, df_name, ax=None):
    y_pred_prob = model_fitted.predict_proba(X)
    y_pred_prob = y_pred_prob[:, 1]
    fpr, tpr, thresholds = sklme.roc_curve(y_true, y_pred_prob, pos_label=pos_label)
    auc = sklme.roc_auc_score(y_true, y_pred_prob).round(3)
    # plt.figure(figsize=(6,4))
    ax.plot(fpr, tpr, label="AUC = %0.3f" % auc)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Base rate\nAUC = 0.5", color="black")
    ax.plot([0, 1], [1, 1], linestyle="--", label="Perfect rate\nAUC = 1", color="red")
    ax.plot([0, 0], [0, 1], linestyle="--", color="red")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Area under the ROC Curve\n(" + df_name + ")")
    ax.legend()


# PR Curve
# @timing__decorator
def my_plot_pr(X, y_true, pos_label, model_fitted, df_name, ax=None):
    y_pred_prob = model_fitted.predict_proba(X)
    y_pred_prob = y_pred_prob[:, 1]
    pr, rc, thresholds = sklme.precision_recall_curve(
        y_true, y_pred_prob, pos_label=pos_label
    )

    auc = sklme.auc(rc, pr).round(3)
    # plt.figure(figsize=(6,4))
    ax.plot(rc, pr, label="PR-AUC = %0.3f" % auc)
    ax.plot(
        [0, 1],
        [0.5, 0.5],
        linestyle="--",
        label="Base rate\nPR-AUC = 0.5",
        color="black",
    )
    ax.plot(
        [1, 0], [1, 1], linestyle="--", label="Perfect rate\nPR-AUC = 1", color="red"
    )
    ax.plot([1, 1], [1, 0], linestyle="--", color="red")
    plt.ylim([-0.05, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Area under the PR Curve\n({df_name})")
    ax.legend()


# Run model with performance report
@timing__decorator
def my_run_model_clf(
    dataset_name, model, X_train, y_train, X_test, y_test, score_and_avg, pos_label
):

    # Khởi tạo model
    my_model = model
    # train model
    my_model = my_model.fit(X_train, y_train)
    # Predict
    y_test_pred = my_model.predict(X_test)
    y_train_pred = my_model.predict(X_train)
    # Evaluate by metrics
    my_model_test_acc = round(sklme.accuracy_score(y_test, y_test_pred), 3)
    my_model_test_recall = round(
        sklme.recall_score(y_test, y_test_pred, average=score_and_avg[1]), 3
    )
    my_model_test_precision = round(
        sklme.precision_score(y_test, y_test_pred, average=score_and_avg[1]), 3
    )
    my_model_test_f1 = round(
        sklme.f1_score(
            y_test,
            y_test_pred,
            average=score_and_avg[1],
        ),
        3,
    )

    # Caculate the Specificity = True Neg / (False Pos + True Neg)
    tn_test, fp_test, fn_test, tp_test = sklme.confusion_matrix(
        y_test, y_test_pred
    ).ravel()
    test_specs = round((tn_test / (tn_test + fp_test)), 3)

    # Caculate the Specificity = True Neg / (False Pos + True Neg)
    tn_train, fp_train, fn_train, tp_train = sklme.confusion_matrix(
        y_train, y_train_pred
    ).ravel()
    train_specs = round((tn_train / (tn_train + fp_train)), 3)

    # Evaluate by metrics: train dataset
    my_model_train_acc = round(sklme.accuracy_score(y_train, y_train_pred), 3)
    my_model_train_recall = round(
        sklme.recall_score(y_train, y_train_pred, average=score_and_avg[1]), 3
    )
    my_model_train_precision = round(
        sklme.precision_score(y_train, y_train_pred, average=score_and_avg[1]), 3
    )
    my_model_train_f1 = round(
        sklme.f1_score(y_train, y_train_pred, average=score_and_avg[1]), 3
    )

    # Present results
    model_name = re.sub(r"\(((.|\n)*)\)", "", str(model))

    mrmclf2 = pd.DataFrame(
        {
            f"Metric ({score_and_avg[1]})": [
                "Accuracy",
                "Precision",
                "Recall/Sensitivity",
                "f1-score",
                "Specificity",
            ],
            "Training Dataset": [
                my_model_train_acc,
                my_model_train_precision,
                my_model_train_recall,
                my_model_train_f1,
                train_specs,
            ],
            "Testing Dataset": [
                my_model_test_acc,
                my_model_test_precision,
                my_model_test_recall,
                my_model_test_f1,
                test_specs,
            ],
            "Diff": [
                my_model_train_acc - my_model_test_acc,
                np.absolute(my_model_train_precision - my_model_test_precision),
                np.absolute(my_model_train_recall - my_model_test_recall),
                np.absolute(my_model_train_f1 - my_model_test_f1),
                np.absolute(train_specs - test_specs),
            ],
        }
    )
    mrmclf2.set_index([f"Metric ({score_and_avg[1]})"], inplace=True)
    mrmclf2.sort_index(ascending=False)
    mrmclf2.columns.names = [f"'{dataset_name}'"]

    display(f"{model_name} Performance")
    display(mrmclf2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    my_plot_conf_mx(y_test, y_test_pred, model_name, "Testing Dataset", ax=ax2)
    my_plot_learning_curve(
        estimator=my_model,
        name=model_name,
        X=X_train,
        y=y_train,
        ax=ax1,
        cv=5,
        scoring=score_and_avg[0],
        plot_name="Training Dataset",
    )
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    my_plot_roc(X_test, y_test, pos_label, my_model, "Testing Dataset", ax=ax3)
    my_plot_pr(X_test, y_test, pos_label, my_model, "Testing Dataset", ax=ax4)

    return mrmclf2, my_model


def my_run_trained_model_clf(
    dataset_name, trained_model, X_test, y_test, score_and_avg, pos_label
):

    # Khởi tạo model
    my_model = trained_model
    # # train model
    # my_model = my_model.fit(X_train, y_train)
    # Predict
    y_test_pred = my_model.predict(X_test)
    # y_train_pred = my_model.predict(X_train)
    # Evaluate by metrics
    my_model_test_acc = round(sklme.accuracy_score(y_test, y_test_pred), 3)
    my_model_test_recall = round(
        sklme.recall_score(y_test, y_test_pred, average=score_and_avg[1]), 3
    )
    my_model_test_precision = round(
        sklme.precision_score(y_test, y_test_pred, average=score_and_avg[1]), 3
    )
    my_model_test_f1 = round(
        sklme.f1_score(
            y_test,
            y_test_pred,
            average=score_and_avg[1],
        ),
        3,
    )

    # Caculate the Specificity = True Neg / (False Pos + True Neg)
    tn_test, fp_test, fn_test, tp_test = sklme.confusion_matrix(
        y_test, y_test_pred
    ).ravel()
    test_specs = round((tn_test / (tn_test + fp_test)), 3)

    # Present results
    model_name = re.sub(r"\(((.|\n)*)\)", "", str(trained_model))

    mrmclf2 = pd.DataFrame(
        {
            f"Metric ({score_and_avg[1]})": [
                "Accuracy",
                "Precision",
                "Recall/Sensitivity",
                "f1-score",
                "Specificity",
            ],
            "Unseen Dataset "
            + "("
            + dataset_name
            + ")": [
                my_model_test_acc,
                my_model_test_precision,
                my_model_test_recall,
                my_model_test_f1,
                test_specs,
            ],
        }
    )
    mrmclf2.set_index([f"Metric ({score_and_avg[1]})"], inplace=True)
    mrmclf2.sort_index(ascending=False)

    display(f"{model_name} Performance")
    display(mrmclf2.T)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    my_plot_conf_mx(y_test, y_test_pred, model_name, "Unseen Dataset", ax=ax)

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    my_plot_roc(X_test, y_test, pos_label, my_model, "Unseen Dataset", ax=ax3)
    my_plot_pr(X_test, y_test, pos_label, my_model, "Unseen Dataset", ax=ax4)

    return mrmclf2, my_model


# KFold Cross Validation
@timing__decorator
def my_kfold_cross_valid(dataset_name, input, output, n_splits, metric, models):
    cv_mean, cv_std, allo = [], [], []
    for k, v in models.items():
        kf = sklms.KFold(n_splits=n_splits)
        cv = sklms.cross_val_score(
            estimator=v,
            X=input,
            y=output,
            scoring=metric,
            cv=kf,
            n_jobs=-1,
        )
        cv_mean.append(round(cv.mean(), 3))
        cv_std.append(round(cv.std(), 3))
        allo.append(v)
    mkcv = pd.DataFrame(
        {
            "Algorithm": allo,
            "Metric": metric,
            "Mean": cv_mean,
            "Errors": cv_std,
        }
    )
    mkcv["Algorithm"] = (
        mkcv["Algorithm"].astype(str).apply(lambda x: re.sub(r"\(((.|\n)*)\)", "", x))
    )
    mkcv = mkcv.sort_values(["Mean"], ascending=False).set_index("Algorithm")
    mkcv["Rank"] = mkcv["Mean"].rank(method="dense", ascending=False).astype("int")

    print(f"\n(+) K-Fold Cross-Validation (dataset: '{dataset_name}')\n")
    return display(mkcv)


# GridSearchCV
@timing__decorator
def my_run_grid_search(
    scaler, estimator, params, cv, scoring_metrics, X_train, y_train
):
    if scaler != "None":
        sc = scaler
        X_train = sc.fit_transform(X_train)
    grid = sklms.GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring=scoring_metrics,
        cv=cv,
        verbose=False,
        n_jobs=-1,
    )
    model_name = str(estimator)
    best = grid.fit(X_train, y_train)
    print("\n", model_name)
    print("-------------------------------")
    print(
        "   Best Score ({}): ".format(scoring_metrics)
        + str(np.round(best.best_score_, 3))
    )
    print("   Best Parameters: ")
    for key, value in grid.best_params_.items():
        print("      , {}= {}".format(key, value))
    return grid.best_params_


# HalvGridSearchCV
@timing__decorator
def my_run_halvgrid_search(
    estimator, params, factor, cv, scoring_metrics, X_train, y_train
):
    grid = sklms.HalvingGridSearchCV(
        estimator=estimator,
        param_grid=params,
        factor=factor,
        scoring=scoring_metrics,
        cv=cv,
        verbose=False,
        random_state=1981,
        n_jobs=-1,
    )
    model_name = str(estimator)
    best = grid.fit(X_train, y_train)
    print("\n", model_name)
    print("-------------------------------")
    print(
        "   Best Score ({}): ".format(scoring_metrics)
        + str(np.round(best.best_score_, 3))
    )
    print("   Best Parameters: ")
    for key, value in grid.best_params_.items():
        print("      , {}= {}".format(key, value))
    return grid.best_params_
