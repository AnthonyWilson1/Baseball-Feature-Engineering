import sys

import numpy as np  # noqa:
import pandas as pd  # noqa:
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff  # noqa:
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from dataset_loader import TestDatasets


def main():
    dataset = TestDatasets()
    # Pull dataset
    all_dataset = dataset.get_test_data_set("mpg")

    # Determine if the response variable is boolean or continuous.
    response_var = all_dataset[2]

    # Dictionary for the categorization of my response/predictors
    cat_dic = {}

    def check_response(data_table, response_col):
        # Need to update for T/F and Yes/No
        if data_table[response_col].value_counts().index.tolist() == [1, 0]:
            cat_dic[response_col] = "Boolean"
        else:
            cat_dic[response_col] = "Continuous"

    # Data table
    dataset_table = all_dataset[0]

    check_response(dataset_table, response_var)

    # Predictor variable array
    predictors = all_dataset[1]

    # Check predictor variables to see if they are cat or cont
    def check_predictors(data_table, col_names):
        for i in col_names:
            for j in data_table[i]:
                if type(j) == str or type(j) == bool:
                    cat_dic[i] = "Discrete"
                else:
                    cat_dic[i] = "Continuous"
                break

    check_predictors(dataset_table, predictors)

    # Generate Plots
    def generate_plots(data_table, col_names, var_dict, response_name):
        if var_dict[response_name] == "Boolean":
            for i in col_names:
                # continuous predictor vs. boolean response
                if cat_dic[i] == "Continuous":
                    group_labels = ["Response = 0", "Response = 1"]  # noqa:
                    pred_0 = data_table[response_name] == 0
                    pred_1 = data_table[response_name] == 1

                    bin_1 = data_table[pred_0]
                    bin_2 = data_table[pred_1]  # noqa:

                    hist_data = [bin_1[i], bin_2[i]]

                    # Create distribution plot with custom bin_size
                    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.5)
                    fig_1.update_layout(
                        title=f"{i} by {response_name}",
                        xaxis_title=f"{i}",
                        yaxis_title="Distribution",
                    )
                    fig_1.show()

                    fig_2 = go.Figure()

                    for curr_hist, curr_group in zip(hist_data, group_labels):
                        fig_2.add_trace(
                            go.Violin(
                                x=np.repeat(curr_group, len(curr_hist)),
                                y=curr_hist,
                                name=curr_group,
                                box_visible=True,  # noqa:
                                meanline_visible=True,  # noqa:
                            )
                        )
                    fig_2.update_layout(
                        title=f"{i} by {response_name}",
                        xaxis_title=f"{response_name}",
                        yaxis_title=f"{i}",
                    )
                    fig_2.show()
                else:
                    # categorical predictor vs. boolean response
                    x_1 = data_table[i].copy().astype("category").cat.codes
                    y_1 = data_table[response_name].copy().astype("category").cat.codes
                    conf_matrix = confusion_matrix(x_1, y_1)
                    fig_no_relationship = go.Figure(
                        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
                    )
                    fig_no_relationship.update_layout(
                        title=f"{i} by {response_name}",
                        xaxis_title=f"{response_name}",
                        yaxis_title=f"{i}",
                    )
                    fig_no_relationship.show()
        else:
            for i in col_names:
                # continuous predictor vs. boolean response
                if cat_dic[i] == "Continuous":
                    fig = px.scatter(
                        x=data_table[i], y=data_table[response_name], trendline="ols"
                    )
                    fig.update_layout(
                        title=f"{i} by {response_name}",
                        xaxis_title=f"{i}",
                        yaxis_title=f"{response_name}",
                    )
                    fig.show()
                else:
                    cat_df = []
                    cat_types = dataset_table[i].unique()
                    for m in cat_types:
                        single_df = data_table[data_table[i] == m]
                        cat_df.append(single_df[response_name])
                    # Create distribution plot with custom bin_size
                    fig_1 = ff.create_distplot(cat_df, cat_types, bin_size=0.2)
                    fig_1.update_layout(
                        title="Continuous Response by Categorical Predictor",
                        xaxis_title="Response",
                        yaxis_title="Distribution",
                    )
                    fig_1.show()

                    fig_2 = go.Figure()
                    for curr_hist, curr_group in zip(cat_df, cat_types):
                        fig_2.add_trace(
                            go.Violin(
                                x=np.repeat(curr_group, len(curr_hist)),
                                y=curr_hist,
                                name=curr_group,
                                box_visible=True,  # noqa:
                                meanline_visible=True,  # noqa:
                            )
                        )
                    fig_2.update_layout(
                        title="Continuous Response by Categorical Predictor",
                        xaxis_title="Groupings",
                        yaxis_title="Response",
                    )
                    fig_2.show()

    # P-value & T-score
    def generate_pval_t(data_table, col_names, var_dict, response_name):
        # calculate p-value & t-score & plots
        print("test")
        ret_df = pd.DataFrame(columns=["Predictors", "P-Value", "T-Value"])
        p_val = []
        t_val = []
        pred = []
        if var_dict[response_name] == "Boolean":
            for i in col_names:
                if cat_dic[i] == "Continuous":
                    pred.append(i)
                    predictor = data_table[i]
                    predictor = statsmodels.api.add_constant(predictor)
                    y = data_table[response_name]
                    feature_name = i
                    linear_regression_model = statsmodels.api.Logit(y, predictor)
                    linear_regression_model_fitted = linear_regression_model.fit()

                    # Get the stats
                    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
                    p_val.append(p_value)
                    t_val.append(t_value)

                    # Plot the figure
                    fig = px.scatter(x=predictor[i], y=y, trendline="ols")
                    fig.update_layout(
                        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                        xaxis_title=f"Variable: {feature_name}",
                        yaxis_title="y",
                    )
                    fig.show()
        else:
            for i in col_names:
                if cat_dic[i] == "Continuous":
                    pred.append(i)
                    predictor = data_table[i]
                    predictor = statsmodels.api.add_constant(predictor)
                    y = data_table[response_name]
                    feature_name = i
                    linear_regression_model = statsmodels.api.OLS(y, predictor)
                    linear_regression_model_fitted = linear_regression_model.fit()

                    # Get the stats
                    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
                    p_val.append(p_value)
                    t_val.append(t_value)

                    # Plot the figure
                    fig = px.scatter(x=predictor[i], y=y, trendline="ols")
                    fig.update_layout(
                        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                        xaxis_title=f"Variable: {feature_name}",
                        yaxis_title="y",
                    )
                    fig.show()
        ret_df["Predictors"] = pred
        ret_df["P-Value"] = p_val
        ret_df["T-Value"] = t_val
        return ret_df

    # Mean and Response Plot and Table Weighted & Unweighted
    def generate_mean_response_bool(data_table, col_names, var_dict, response_name):
        mean_res_ranking = {}
        mean_res_ranking_w = {}
        for i in col_names:
            if var_dict[i] == "Continuous":
                # Histogram
                fig = go.Figure()
                counts, bins = np.histogram(data_table[i], bins=10)
                bins2 = 0.5 * (bins[:-1] + bins[1:])
                trace0 = go.Bar(x=bins2, y=counts, name=i, yaxis="y2", opacity=0.5)
                # Plot Class Average
                class_avg = [
                    data_table[response_name].value_counts(normalize=True)[1]
                ] * 10
                trace1 = go.Scatter(x=bins2, y=class_avg, name=rf"$\mu_{{{i}}}$")
                # Plot Response Average
                bin_avg = []

                j = 0
                k = 1

                for m in range(0, 10):

                    df_b = data_table[
                        (data_table[i] >= bins[j]) & (data_table[i] < bins[k])
                    ]

                    try:
                        bin_avg.append(
                            df_b[response_name].value_counts(normalize=True)[1]
                        )
                    except KeyError:
                        bin_avg.append(0)

                    j += 1
                    k += 1

                trace2 = go.Scatter(
                    x=bins2, y=bin_avg, name=rf"$\mu_{{response_name}}-\mu_{{{i}}}$"
                )

                fig.update_layout(
                    legend=dict(orientation="v"),
                    yaxis=dict(
                        title=dict(text="Response"),
                        side="left",
                        range=[0, max(data_table[response_name])],
                    ),
                    yaxis2=dict(
                        title=dict(text=i),
                        side="right",
                        range=[0, max(counts) + 1],
                        overlaying="y",
                        tickmode="sync",
                    ),
                )
                # Add figure title
                fig.update_layout(title_text=response_name)

                # Set x-axis title
                fig.update_xaxes(title_text="Predictor Bin")

                # Show plots
                fig.add_trace(trace0)
                fig.add_trace(trace1)
                fig.add_trace(trace2)
                fig.show()

                # Table
                diff_mean_col = [
                    "i",
                    "LowerBin",
                    "UpperBin",
                    "BinCenters",
                    "BinCounts",
                    "BinMeans",
                    "PredictorMean",
                    "MeanSquareDiff",
                    "PopulationProportion",
                    "MeanSquaredDiffWeighted",
                ]
                df_diff_mean = pd.DataFrame(columns=diff_mean_col)
                df_diff_mean["i"] = [j for j in range(0, 10)]
                df_diff_mean["LowerBin"] = bins[:-1]
                df_diff_mean["UpperBin"] = bins[1:]
                df_diff_mean["BinCenters"] = (
                    df_diff_mean["LowerBin"] + df_diff_mean["UpperBin"]
                ) / 2
                df_diff_mean["BinCounts"] = counts
                df_diff_mean["BinMeans"] = bin_avg
                df_diff_mean["PredictorMean"] = class_avg
                df_diff_mean["MeanSquareDiff"] = pow(
                    df_diff_mean["BinMeans"] - df_diff_mean["PredictorMean"], 2
                )
                df_diff_mean["PopulationProportion"] = df_diff_mean["BinCounts"] / sum(
                    df_diff_mean["BinCounts"]
                )
                df_diff_mean["MeanSquaredDiffWeighted"] = (
                    df_diff_mean["PopulationProportion"]
                    * df_diff_mean["MeanSquareDiff"]
                )
                mean_res_ranking[i] = round(sum(df_diff_mean["MeanSquareDiff"]), 3)
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"]), 3
                )
            else:
                # Histogram
                fig = px.histogram(data_table[response_name], data_table[i])
                # Plot Class Average
                cat_types = dataset_table[i].unique()
                class_avg = [
                    data_table[response_name].value_counts(normalize=True)[1]
                ] * len(cat_types)
                trace1 = go.Scatter(x=cat_types, y=class_avg, name=rf"$\mu_{{{i}}}$")

                # Plot Response Average
                bin_avg = []
                counts = []
                for m in cat_types:
                    fil_tab = dataset_table[dataset_table[i] == m]
                    counts.append(len(fil_tab))
                    try:
                        bin_avg.append(
                            fil_tab[response_name].value_counts(normalize=True)[1]
                        )
                    except KeyError:
                        bin_avg.append(0)
                trace2 = go.Scatter(
                    x=cat_types, y=bin_avg, name=rf"$\mu_{{response_name}}-\mu_{{{i}}}$"
                )

                fig.update_layout(
                    legend=dict(orientation="v"),
                    yaxis=dict(
                        title=dict(text="Response"),
                        side="left",
                        range=[0, max(data_table[response_name])],
                    ),
                    yaxis2=dict(
                        title=dict(text=i),
                        side="right",
                        range=[0, len(cat_types) + 1],
                        overlaying="y",
                        tickmode="sync",
                    ),
                )
                # Add figure title
                fig.update_layout(title_text=response_name)

                # Set x-axis title
                fig.update_xaxes(title_text="Predictor Bin")

                # Show plots
                fig.add_trace(trace1)
                fig.add_trace(trace2)
                fig.show()

                # Table
                diff_mean_col = [
                    "i",
                    "LowerBin",
                    "UpperBin",
                    "BinCenters",
                    "BinCounts",
                    "BinMeans",
                    "PredictorMean",
                    "MeanSquareDiff",
                    "PopulationProportion",
                    "MeanSquaredDiffWeighted",
                ]
                df_diff_mean = pd.DataFrame(columns=diff_mean_col)
                df_diff_mean["i"] = [j for j in range(0, len(cat_types))]
                df_diff_mean["LowerBin"] = cat_types
                df_diff_mean["UpperBin"] = cat_types
                df_diff_mean["BinCenters"] = cat_types
                df_diff_mean["BinCounts"] = counts
                df_diff_mean["BinMeans"] = bin_avg
                df_diff_mean["PredictorMean"] = class_avg
                df_diff_mean["MeanSquareDiff"] = pow(
                    df_diff_mean["BinMeans"] - df_diff_mean["PredictorMean"], 2
                )
                df_diff_mean["PopulationProportion"] = df_diff_mean["BinCounts"] / sum(
                    df_diff_mean["BinCounts"]
                )
                df_diff_mean["MeanSquaredDiffWeighted"] = (
                    df_diff_mean["PopulationProportion"]
                    * df_diff_mean["MeanSquareDiff"]
                )
                mean_res_ranking[i] = round(sum(df_diff_mean["MeanSquareDiff"]), 3)
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"]), 3
                )
        # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
        df_mean_r = pd.DataFrame(
            mean_res_ranking.items(), columns=["Predictors", "Mean of Response"]
        )
        df_mean_w = pd.DataFrame(
            mean_res_ranking_w.items(),
            columns=["Predictors", "Mean of Response Weighted"],
        )
        final_df = pd.merge(df_mean_w, df_mean_r, on="Predictors")
        return final_df

    def generate_mean_response_cont(data_table, col_names, var_dict, response_name):
        mean_res_ranking = {}
        mean_res_ranking_w = {}
        for i in col_names:
            if var_dict[i] == "Continuous":
                # Histogram
                fig = go.Figure()
                counts, bins = np.histogram(data_table[i], bins=10)
                bins2 = 0.5 * (bins[:-1] + bins[1:])
                trace0 = go.Bar(x=bins2, y=counts, name=i, yaxis="y2", opacity=0.5)
                # Plot Class Average
                class_avg = [data_table[response_name].mean()] * 10
                trace1 = go.Scatter(x=bins2, y=class_avg, name=rf"$\mu_{{{i}}}$")
                # Plot Response Average
                bin_avg = []

                j = 0
                k = 1

                for m in range(0, 10):

                    df_b = data_table[
                        (data_table[i] >= bins[j]) & (data_table[i] < bins[k])
                    ]

                    try:
                        bin_avg.append(df_b[response_name].mean())
                    except KeyError:
                        bin_avg.append(0)

                    j += 1
                    k += 1

                trace2 = go.Scatter(
                    x=bins2, y=bin_avg, name=rf"$\mu_{{response_name}}-\mu_{{{i}}}$"
                )

                fig.update_layout(
                    legend=dict(orientation="v"),
                    yaxis=dict(
                        title=dict(text="Response"),
                        side="left",
                        range=[0, max(data_table[response_name])],
                    ),
                    yaxis2=dict(
                        title=dict(text=i),
                        side="right",
                        range=[0, max(counts) + 1],
                        overlaying="y",
                        tickmode="sync",
                    ),
                )
                # Add figure title
                fig.update_layout(title_text=response_name)

                # Set x-axis title
                fig.update_xaxes(title_text="Predictor Bin")

                # Show plots
                fig.add_trace(trace0)
                fig.add_trace(trace1)
                fig.add_trace(trace2)
                fig.show()

                # Table
                diff_mean_col = [
                    "i",
                    "LowerBin",
                    "UpperBin",
                    "BinCenters",
                    "BinCounts",
                    "BinMeans",
                    "PredictorMean",
                    "MeanSquareDiff",
                    "PopulationProportion",
                    "MeanSquaredDiffWeighted",
                ]
                df_diff_mean = pd.DataFrame(columns=diff_mean_col)
                df_diff_mean["i"] = [j for j in range(0, 10)]
                df_diff_mean["LowerBin"] = bins[:-1]
                df_diff_mean["UpperBin"] = bins[1:]
                df_diff_mean["BinCenters"] = (
                    df_diff_mean["LowerBin"] + df_diff_mean["UpperBin"]
                ) / 2
                df_diff_mean["BinCounts"] = counts
                df_diff_mean["BinMeans"] = bin_avg
                df_diff_mean["PredictorMean"] = class_avg
                df_diff_mean["MeanSquareDiff"] = pow(
                    df_diff_mean["BinMeans"] - df_diff_mean["PredictorMean"], 2
                )
                df_diff_mean["PopulationProportion"] = df_diff_mean["BinCounts"] / sum(
                    df_diff_mean["BinCounts"]
                )
                df_diff_mean["MeanSquaredDiffWeighted"] = (
                    df_diff_mean["PopulationProportion"]
                    * df_diff_mean["MeanSquareDiff"]
                )
                mean_res_ranking[i] = round(sum(df_diff_mean["MeanSquareDiff"]), 3)
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"]), 3
                )
            else:
                # Histogram
                fig = px.histogram(data_table[response_name], data_table[i])
                # Plot Class Average
                cat_types = dataset_table[i].unique()
                class_avg = [data_table[response_name].mean()] * len(cat_types)
                trace1 = go.Scatter(x=cat_types, y=class_avg, name=rf"$\mu_{{{i}}}$")

                # Plot Response Average
                bin_avg = []
                counts = []
                for m in cat_types:
                    fil_tab = dataset_table[dataset_table[i] == m]
                    counts.append(len(fil_tab))
                    try:
                        bin_avg.append(fil_tab[response_name].mean())
                    except KeyError:
                        bin_avg.append(0)
                trace2 = go.Scatter(
                    x=cat_types, y=bin_avg, name=rf"$\mu_{{response_name}}-\mu_{{{i}}}$"
                )

                fig.update_layout(
                    legend=dict(orientation="v"),
                    yaxis=dict(
                        title=dict(text="Response"),
                        side="left",
                        range=[0, max(data_table[response_name])],
                    ),
                    yaxis2=dict(
                        title=dict(text=i),
                        side="right",
                        range=[0, len(cat_types) + 1],
                        overlaying="y",
                        tickmode="sync",
                    ),
                )
                # Add figure title
                fig.update_layout(title_text=response_name)

                # Set x-axis title
                fig.update_xaxes(title_text="Predictor Bin")

                # Show plots
                fig.add_trace(trace1)
                fig.add_trace(trace2)
                fig.show()

                # Table
                diff_mean_col = [
                    "i",
                    "LowerBin",
                    "UpperBin",
                    "BinCenters",
                    "BinCounts",
                    "BinMeans",
                    "PredictorMean",
                    "MeanSquareDiff",
                    "PopulationProportion",
                    "MeanSquaredDiffWeighted",
                ]
                df_diff_mean = pd.DataFrame(columns=diff_mean_col)
                df_diff_mean["i"] = [j for j in range(0, len(cat_types))]
                df_diff_mean["LowerBin"] = cat_types
                df_diff_mean["UpperBin"] = cat_types
                df_diff_mean["BinCenters"] = cat_types
                df_diff_mean["BinCounts"] = counts
                df_diff_mean["BinMeans"] = bin_avg
                df_diff_mean["PredictorMean"] = class_avg
                df_diff_mean["MeanSquareDiff"] = pow(
                    df_diff_mean["BinMeans"] - df_diff_mean["PredictorMean"], 2
                )
                df_diff_mean["PopulationProportion"] = df_diff_mean["BinCounts"] / sum(
                    df_diff_mean["BinCounts"]
                )
                df_diff_mean["MeanSquaredDiffWeighted"] = (
                    df_diff_mean["PopulationProportion"]
                    * df_diff_mean["MeanSquareDiff"]
                )
                mean_res_ranking[i] = round(sum(df_diff_mean["MeanSquareDiff"]), 3)
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"]), 3
                )
        # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
        df_mean_r = pd.DataFrame(
            mean_res_ranking.items(), columns=["Predictors", "Mean of Response"]
        )
        df_mean_w = pd.DataFrame(
            mean_res_ranking_w.items(),
            columns=["Predictors", "Mean of Response Weighted"],
        )
        final_df = pd.merge(df_mean_w, df_mean_r, on="Predictors")
        return final_df

    def generate_random_forrest(data_table, col_names, var_dict, response_name):
        # Get continuous pred
        cont_col = []
        for i in col_names:
            if var_dict[i] == "Continuous":
                cont_col.append(i)
        # Fit model (https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
        forest = RandomForestClassifier(random_state=0)
        forest.fit(data_table[cont_col], data_table[response_name])
        importance = forest.feature_importances_
        df_rank = pd.DataFrame(columns=["Predictors", "Random Forrest Importance Rank"])
        df_rank["Predictors"] = cont_col
        df_rank["Random Forrest Importance Rank"] = importance
        return df_rank

    # Generate HTML
    def generate_html(data_table, col_names, var_dict, response_name):  # noqa:
        if var_dict[response_name] == "Boolean":
            generate_plots(data_table, col_names, var_dict, response_name)
            p_val_t = generate_pval_t(
                dataset_table, predictors, cat_dic, response_var
            ).to_html()
            mean_r = generate_mean_response_bool(
                dataset_table, predictors, cat_dic, response_var
            ).to_html()
            rf_imp = generate_random_forrest(
                dataset_table, predictors, cat_dic, response_var
            ).to_html()
        else:
            generate_plots(data_table, col_names, var_dict, response_name)
            p_val_t = generate_pval_t(
                dataset_table, predictors, cat_dic, response_var
            ).to_html()
            mean_r = generate_mean_response_cont(
                dataset_table, predictors, cat_dic, response_var
            ).to_html()
            rf_imp = "N/A For Continuous Response"
        return p_val_t, mean_r, rf_imp

    p_value_t_value_table, mean_response_table, rf_imp_table = generate_html(
        dataset_table, predictors, cat_dic, response_var
    )

    print(p_value_t_value_table)
    print(mean_response_table)
    print(rf_imp_table)


if __name__ == "__main__":
    sys.exit(main())
