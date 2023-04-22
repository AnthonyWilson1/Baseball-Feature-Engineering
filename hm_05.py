import math
import sys
import warnings  # noqa:
from pathlib import Path

import numpy as np  # noqa:
import pandas as pd  # noqa:
import plotly.io
import sqlalchemy
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff  # noqa:
from plotly import graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy import text


def main():
    # Create Plots Folder
    home = Path.cwd()
    Path(f"{home}/Plots").mkdir(parents=True, exist_ok=True)

    # Set pandas to display all columns and rows
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    db_user = "test"
    db_pass = "test"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    connection = sql_engine.connect()

    index_game_id = text("CREATE INDEX idx_game_id ON team_batting_counts (game_id)")

    connection.execute(index_game_id)

    index_game_id = text("CREATE INDEX idx_team_id ON team_batting_counts (team_id)")

    connection.execute(index_game_id)

    # response is home team wins (0,1)
    # each row is going to be a game a team plays
    # have two teams and the response is that the home team wins (0,1)
    # team_results then filter for home_games
    # each row will be one game with team level stats before that point

    # Filter Response
    query_response = text(
        "SELECT game_id AS game_id_orig, team_id, opponent_id, "
        "home_away, win_lose FROM team_results WHERE home_away = 'H'"
    )
    df_response = pd.read_sql_query(query_response, connection)
    #
    # Filter First Feature 100 day rolling average home runs
    # Join Game and Team Batter Counts in a temp table to get the dates,
    # then join the table with itself so home_team and away_team are on the same row
    # get the 100 day rolling average by merging with itself
    # Home_Run, team_id, opponent_team_id, homeTeam, awayTeam

    # Team Batting Stats

    temp_team_counts_query = text(
        """CREATE TEMPORARY TABLE team_batting_counts_dates AS
        (SELECT tc.game_id AS game_id_tc, gam.local_date AS game_date,
        tc.team_id AS team_tc, tc.Home_Run AS Home_Run_tc,
        tc.opponent_team_id AS opponent_team_id_tc,
        tc.homeTeam AS homeTeam_tc, tc.awayTeam as awayTeam_tc,
        tc.atBat AS atBat_tc, tc.Hit AS Hit_tc
        FROM team_batting_counts tc
        JOIN game gam ON tc.game_id = gam.game_id)"""
    )

    temp_home_away_same = text(
        """CREATE TEMPORARY TABLE team_batting_counts_away_same AS
        (SELECT orig.game_id_tc AS game_id_orig, copy.game_id_tc AS game_id_copy,
        orig.game_date AS game_date_orig, orig.team_tc AS team1,
        copy.team_tc AS team2, orig.Home_Run_tc AS Home_Run_t1,
        copy.Home_Run_tc AS Home_Run_t2,
        orig.atBat_tc AS atBat_t1, copy.atBat_tc AS atBat_t2,
        orig.Hit_tc AS Hit_t1, copy.Hit_tc AS Hit_t2
        FROM team_batting_counts_dates orig
        JOIN team_batting_counts_dates copy ON orig.game_id_tc = copy.game_id_tc
        WHERE orig.team_tc != copy.team_tc
        )"""
    )

    # temp_home_run_final = text('''CREATE TEMPORARY TABLE team_home_run AS
    #     (SELECT orig.game_id_orig as final_game_id,
    #     )''')

    connection.execute(temp_team_counts_query)
    # pull_temp_query = text("SELECT * FROM team_batting_counts_dates")
    # result = pd.read_sql_query(pull_temp_query, connection)

    connection.execute(temp_home_away_same)
    pull_temp_query2 = text("SELECT * FROM team_batting_counts_away_same")
    result2 = pd.read_sql_query(pull_temp_query2, connection)

    team_batting_counts_no_dup = result2.drop_duplicates(
        subset="game_id_orig", keep="last"
    )

    team_batting_counts_no_dup.to_sql(
        "team_batting_counts_no_dup", connection, if_exists="replace", index=False
    )

    temp_rolling_atbat = text(
        """ SELECT # noqa:
        table_one.game_id_orig as game_id_orig
        ,table_one.game_date_orig as game_date_start
        ,table_two.game_date_orig as game_date_end
        ,table_one.team1 as team1
        ,table_one.team2 as team2
        ,table_one.Home_Run_t1 as Home_Run_t1
        ,table_one.Home_Run_t2 as Home_Run_t2
        ,SUM(table_two.Hit_t1) as Hits_t1
        ,SUM(table_two.atBat_t1) as atBat_t1
        ,SUM(table_two.Hit_t2) as Hits_t2
        ,SUM(table_two.atBat_t2) as atBat_t2
        ,IF (SUM(table_two.atBat_t1)>0, SUM(table_two.Hit_t1) / SUM(table_two.atBat_t1), 0) AS RollingBattingAverageT1
        ,IF (SUM(table_two.atBat_t2)>0, SUM(table_two.Hit_t2) / SUM(table_two.atBat_t2), 0) AS RollingBattingAverageT2
        FROM team_batting_counts_no_dup AS table_one
        LEFT JOIN team_batting_counts_no_dup AS table_two
        ON table_one.team1 = table_two.team1
        AND table_two.game_date_orig >= DATE_SUB(table_one.game_date_orig,INTERVAL 100 DAY)
        AND table_two.game_date_orig < table_one.game_date_orig
        GROUP BY team1, game_date_start, game_id_orig
    """
    )

    final = pd.read_sql_query(temp_rolling_atbat, connection)

    # print(final.head())

    # Pitching Stats

    df_pitcher = pd.read_sql_query(  # noqa:
        text(
            """SELECT game_id AS game_id_orig, season_avg, career_w, season_w, season_hr, career_hr, season_so,
         career_so,
        career_l, season_l, loaded_avg FROM pitcher_stat"""
        ),
        connection,
    )  # noqa:

    features = pd.merge(final, df_pitcher, on="game_id_orig", how="outer")

    features_f = features.dropna()

    final_table = pd.merge(features_f, df_response, on="game_id_orig", how="outer")

    final_table_f = final_table.dropna()

    # print(final_table_f.head())

    final_table_f["win_lose"] = final_table_f["win_lose"].replace({"W": 1, "L": 0})

    predictors = [
        "Home_Run_t1",
        "Home_Run_t2",
        "RollingBattingAverageT1",
        "RollingBattingAverageT2",
        "season_avg",
        "career_w",
        "season_w",
        "season_hr",
        "career_hr",
        "season_so",
        "career_so",
        "career_l",
        "season_l",
    ]
    response_var = "win_lose"

    # Dictionary for the categorization of my response/predictors
    cat_dic = {}

    def check_response(data_table, response_col):
        # Need to update for T/F and Yes/No
        if data_table[response_col].value_counts().index.tolist() == [1, 0]:
            cat_dic[response_col] = "Boolean"
        else:
            cat_dic[response_col] = "Continuous"

    check_response(final_table_f, response_var)

    def check_predictors(data_table, col_names):
        for i in col_names:
            for j in data_table[i]:
                if type(j) == str or type(j) == bool:
                    cat_dic[i] = "Categorical"
                else:
                    cat_dic[i] = "Continuous"
                break

    check_predictors(final_table_f, predictors)

    def generate_plots_m(data_table, col_names, var_dict, response_name):
        # plot_df = pd.DataFrame(columns=["Feature", "Plot", "Mean of Response Plot", "Diff Mean Response (Weighted)",
        #                                 "Diff Mean Response (Unweighted)", "Random Forest Variable Importance",
        #                                 "P-Value", "T-Score"])
        plot_df = pd.DataFrame(columns=["Feature", "Plot"])
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
                    #
                    # # Create distribution plot with custom bin_size
                    # fig_1 = ff.create_distplot(hist_data, group_labels)
                    # fig_1.update_layout(
                    #     title=f"{i} by {response_name}",
                    #     xaxis_title=f"{i}",
                    #     yaxis_title="Distribution",
                    #     bargap=0
                    # )
                    #
                    # fig_1.write_html(file=f"Plots/{i}_vs_{response_name}.html")
                    #
                    # plot_df.loc[len(plot_df)] = [i, f"Plots/{i}_vs_{response_name}.html", "N/A", "N/A", "N/A", "N/A",
                    #                              "N/A", "N/A"]

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

                    fig_2.write_html(file=f"Plots/{i}_vs_{response_name}.html")

                    # plot_df.loc[len(plot_df)] = [i, f"Plots/{i}_vs_{response_name}.html", "N/A", "N/A", "N/A", "N/A",
                    #                              "N/A", "N/A"]
                    plot_df.loc[len(plot_df)] = [
                        i,
                        f"Plots/{i}_vs_{response_name}.html",
                    ]

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
                    fig_no_relationship.write_html(
                        file=f"Plots/{i}_vs_{response_name}.html"
                    )

                    # plot_df.loc[len(plot_df)] = [i, f"Plots/{i}_vs_{response_name}.html", "N/A", "N/A", "N/A", "N/A",
                    #                              "N/A", "N/A"]

                    plot_df.loc[len(plot_df)] = [
                        i,
                        f"Plots/{i}_vs_{response_name}.html",
                    ]
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
                    fig.write_html(file=f"Plots/{i}_vs_{response_name}.html")

                    plot_df.loc[len(plot_df)] = [
                        i,
                        f"Plots/{i}_vs_{response_name}.html",
                    ]
                else:
                    cat_df = []
                    cat_types = data_table[i].unique()
                    for m in cat_types:
                        single_df = data_table[data_table[i] == m]
                        cat_df.append(single_df[response_name])
                    #     # Create distribution plot with custom bin_size
                    #     fig_1 = ff.create_distplot(cat_df, cat_types, bin_size=0.2)
                    #     fig_1.update_layout(
                    #         title="Continuous Response by Categorical Predictor",
                    #         xaxis_title="Response",
                    #         yaxis_title="Distribution",
                    #     )
                    #     fig_1.write_html(file=f"Plots/{i}_vs_{response_name}.html")
                    #
                    #     plot_df.loc[len(plot_df)] = [i, f"Plots/{i}_vs_{response_name}.html", "N/A", "N/A", "N/A",
                    #     "N/A",
                    #                                  "N/A", "N/A"]

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
                    fig_2.write_html(file=f"Plots/{i}_vs_{response_name}.html")

                    # plot_df.loc[len(plot_df)] = [i, f"Plots/{i}_vs_{response_name}.html", "N/A", "N/A", "N/A", "N/A",
                    #                              "N/A", "N/A"]
                    plot_df.loc[len(plot_df)] = [
                        i,
                        f"Plots/{i}_vs_{response_name}.html",
                    ]

        return plot_df

    def generate_pval_t_m(data_table, col_names, var_dict, response_name):
        # calculate p-value & t-score & plots
        # ret_df = pd.DataFrame(columns=["Feature", "Plot", "Mean of Response Plot", "Diff Mean Response (Weighted)",
        #                                "Diff Mean Response (Unweighted)", "Random Forest Variable Importance",
        #                                "P-Value", "T-Score"])
        ret_df = pd.DataFrame(columns=["Feature", "P-Value", "T-Score"])
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
                    feature_name = i  # noqa:
                    linear_regression_model = statsmodels.api.Logit(y, predictor)
                    linear_regression_model_fitted = linear_regression_model.fit()

                    # Get the stats
                    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
                    p_val.append(p_value)
                    t_val.append(t_value)

                    # Plot the figure
                    # fig = px.scatter(x=predictor[i], y=y, trendline="ols")
                    # fig.update_layout(
                    #     title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                    #     xaxis_title=f"Variable: {feature_name}",
                    #     yaxis_title="y",
                    # )
                    # fig.show()
        else:
            for i in col_names:
                if cat_dic[i] == "Continuous":
                    pred.append(i)
                    predictor = data_table[i]
                    predictor = statsmodels.api.add_constant(predictor)
                    y = data_table[response_name]
                    feature_name = i  # noqa:
                    linear_regression_model = statsmodels.api.OLS(y, predictor)
                    linear_regression_model_fitted = linear_regression_model.fit()

                    # Get the stats
                    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
                    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
                    p_val.append(p_value)
                    t_val.append(t_value)

                    # Plot the figure
                    # fig = px.scatter(x=predictor[i], y=y, trendline="ols")
                    # fig.update_layout(
                    #     title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                    #     xaxis_title=f"Variable: {feature_name}",
                    #     yaxis_title="y",
                    # )
                    # fig.show()
        ret_df["Feature"] = pred
        ret_df["P-Value"] = p_val
        ret_df["T-Score"] = t_val
        return ret_df

    def generate_mean_response_bool_m(data_table, col_names, var_dict, response_name):
        mean_res_ranking = {}
        mean_res_ranking_w = {}
        plots = []
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
                        (data_table[i] >= bins[j]) & (data_table[i] <= bins[k])
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
                fig.write_html(file=f"Plots/{i}_vs_{response_name}_mr.html")
                plots.append(f"Plots/{i}_vs_{response_name}_mr.html")

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
                mean_res_ranking[i] = round(sum(df_diff_mean["MeanSquareDiff"]) / 10, 6)
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"] / 10), 6
                )
            else:
                # Histogram (Updated)
                fig = px.histogram(data_table[response_name], data_table[i])
                # Plot Class Average
                cat_types = data_table[i].unique()
                class_avg = [
                    data_table[response_name].value_counts(normalize=True)[1]
                ] * len(cat_types)
                trace1 = go.Scatter(x=cat_types, y=class_avg, name=rf"$\mu_{{{i}}}$")

                # Plot Response Average
                bin_avg = []
                counts = []
                for m in cat_types:
                    fil_tab = data_table[data_table[i] == m]
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
                fig.write_html(file=f"Plots/{i}_vs_{response_name}_mr.html")
                plots.append(f"Plots/{i}_vs_{response_name}_mr.html")

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
                mean_res_ranking[i] = round(
                    sum(df_diff_mean["MeanSquareDiff"]) / len(cat_types), 6
                )
                mean_res_ranking_w[i] = round(
                    sum(df_diff_mean["MeanSquaredDiffWeighted"]) / len(cat_types), 6
                )
        # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
        df_mean_r = pd.DataFrame(
            mean_res_ranking.items(), columns=["Feature", "Mean of Response"]
        )
        df_mean_w = pd.DataFrame(
            mean_res_ranking_w.items(),
            columns=["Feature", "Mean of Response Weighted"],
        )
        final_df = pd.merge(df_mean_w, df_mean_r, on="Feature")
        final_df["Mean of Response Plot"] = plots
        return final_df, mean_res_ranking, mean_res_ranking_w

    def generate_random_forrest_m(data_table, col_names, var_dict, response_name):
        # Get continuous pred
        cont_col = []
        for i in col_names:
            if var_dict[i] == "Continuous":
                cont_col.append(i)
        # Fit model (https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
        forest = RandomForestClassifier(random_state=0)
        forest.fit(data_table[cont_col], data_table[response_name])
        importance = forest.feature_importances_
        # df_rank = pd.DataFrame(columns=["Feature", "Plot", "Mean of Response Plot", "Diff Mean Response (Weighted)",
        #                                 "Diff Mean Response (Unweighted)", "Random Forest Variable Importance",
        #                                 "P-Value", "T-Score"])
        df_rank = pd.DataFrame(columns=["Feature", "Random Forest Variable Importance"])
        df_rank["Feature"] = cont_col
        df_rank["Random Forest Variable Importance"] = importance
        return df_rank

    def generate_html_m(data_table, col_names, var_dict, response_name):  # noqa:
        if var_dict[response_name] == "Boolean":
            plots_table_f = generate_plots_m(
                data_table, col_names, var_dict, response_name
            )
            p_val_t = generate_pval_t_m(data_table, predictors, cat_dic, response_var)
            mean_r, w, u = generate_mean_response_bool_m(
                data_table, predictors, cat_dic, response_var
            )
            rf_imp = generate_random_forrest_m(
                data_table, predictors, cat_dic, response_var
            )
        else:
            plots_table_f = generate_plots_m(
                data_table, col_names, var_dict, response_name
            )
            p_val_t = generate_pval_t_m(data_table, predictors, cat_dic, response_var)
            mean_r, w, u = generate_mean_response_cont_m(  # noqa:
                data_table, predictors, cat_dic, response_var
            )
            rf_imp = "N/A For Continuous Response"
        return p_val_t, mean_r, rf_imp, plots_table_f

        # correlation for cat/cat tschuprow's T and Cramer's V

    def fill_na(data):
        if isinstance(data, pd.Series):
            return data.fillna(0)
        else:
            return np.array([value if value is not None else 0 for value in data])

    # def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    #     """
    #     Calculates correlation statistic for categorical-categorical association.
    #     The two measures supported are:
    #     1. Cramer'V ( default )
    #     2. Tschuprow'T
    #
    #     SOURCES:
    #     1.) CODE: https://github.com/MavericksDS/pycorr
    #     2.) Used logic from:
    #         https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    #         to ignore yates correction factor on 2x2
    #     3.) Haven't validated Tschuprow
    #
    #     Bias correction and formula's taken from : https://www.researchgate.net/publication/270277061_
    #     A_bias-correction_for_Cramer's_V_and_Tschuprow's_T
    #
    #     Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    #     Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    #     Parameters:
    #     -----------
    #     x : list / ndarray / Pandas Series
    #         A sequence of categorical measurements
    #     y : list / NumPy ndarray / Pandas Series
    #         A sequence of categorical measurements
    #     bias_correction : Boolean, default = True
    #     tschuprow : Boolean, default = False
    #                For choosing Tschuprow as measure
    #     Returns:
    #     --------
    #     float in the range of [0,1]
    #     """
    #     corr_coeff = np.nan
    #     try:
    #         x, y = fill_na(x), fill_na(y)
    #         crosstab_matrix = pd.crosstab(x, y)
    #         n_observations = crosstab_matrix.sum().sum()
    #
    #         yates_correct = True
    #         if bias_correction:
    #             if crosstab_matrix.shape == (2, 2):
    #                 yates_correct = False
    #
    #         chi2, _, _, _ = stats.chi2_contingency(
    #             crosstab_matrix, correction=yates_correct
    #         )
    #         phi2 = chi2 / n_observations
    #
    #         # r and c are number of categories of x and y
    #         r, c = crosstab_matrix.shape
    #         if bias_correction:
    #             phi2_corrected = max(
    #                 0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
    #             )
    #             r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
    #             c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
    #             if tschuprow:
    #                 corr_coeff = np.sqrt(
    #                     phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
    #                 )
    #                 return corr_coeff
    #             corr_coeff = np.sqrt(
    #                 phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
    #             )
    #             return corr_coeff
    #         if tschuprow:
    #             corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
    #             return corr_coeff
    #         corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
    #         return corr_coeff
    #     except Exception as ex:  # noqa:
    #         if tschuprow:
    #             warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
    #         else:
    #             warnings.warn("Error calculating Cramer's V", RuntimeWarning)
    #         return corr_coeff
    #
    # def cat_cont_correlation_ratio(categories, values):
    #     """
    #     Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    #     SOURCE:
    #     1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    #     :param categories: Numpy array of categories
    #     :param values: Numpy array of values
    #     :return: correlation
    #     """
    #     f_cat, _ = pd.factorize(categories)
    #     cat_num = np.max(f_cat) + 1
    #     y_avg_array = np.zeros(cat_num)
    #     n_array = np.zeros(cat_num)
    #     for i in range(0, cat_num):
    #         cat_measures = values[np.argwhere(f_cat == i).flatten()]
    #         n_array[i] = len(cat_measures)
    #         y_avg_array[i] = np.average(cat_measures)
    #     y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    #     numerator = np.sum(
    #         np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    #     )
    #     denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    #     if numerator == 0:
    #         eta = 0.0
    #     else:
    #         eta = np.sqrt(numerator / denominator)
    #     return eta
    #
    # def cat_cat_matrix(data_table, col_names, var_dict, plot, mrplot):
    #
    #     # corr table
    #
    #     cor_tab_c = pd.DataFrame(
    #         columns=["cat_1", "cat_2", "corr", "cat_1_url", "cat_2_url"]
    #     )
    #     cor_tab_t = pd.DataFrame(
    #         columns=["cat_1", "cat_2", "corr", "cat_1_url", "cat_2_url"]
    #     )
    #
    #     # cat col array
    #     cat_col = []
    #     for i in col_names:
    #         if var_dict[i] == "Categorical":
    #             cat_col.append(i)
    #
    #     # n by n matrix
    #     zeros_array_cramer = np.zeros((len(cat_col), len(cat_col)))
    #     zeros_array_tsch = np.zeros((len(cat_col), len(cat_col)))
    #
    #     # loop through each cat with itself and fill in matrix
    #     for num, cat in enumerate(cat_col):
    #         for num2, cat2 in enumerate(cat_col):
    #             cat_1 = fill_na(data_table[cat])
    #             cat_2 = fill_na(data_table[cat2])
    #             zeros_array_cramer[num][num2] = cat_correlation(cat_1, cat_2)
    #             zeros_array_tsch[num][num2] = cat_correlation(
    #                 cat_1, cat_2, tschuprow=True
    #             )
    #             cor_tab_c.loc[len(cor_tab_c)] = [
    #                 cat,
    #                 cat2,
    #                 cat_correlation(cat_1, cat_2),
    #                 plot[cat],
    #                 mrplot[cat2],
    #             ]
    #             cor_tab_t.loc[len(cor_tab_t)] = [
    #                 cat,
    #                 cat2,
    #                 cat_correlation(cat_1, cat_2, tschuprow=True),
    #                 plot[cat],
    #                 mrplot[cat2],
    #             ]
    #
    #     # https://en.ai-research-collection.com/plotly-heatmap/
    #
    #     x = cat_col
    #     y = cat_col
    #     z = zeros_array_cramer
    #     z2 = zeros_array_tsch
    #
    #     fig = ff.create_annotated_heatmap(
    #         z,
    #         x=x,
    #         y=y,
    #         annotation_text=np.around(z, decimals=4),
    #         hoverinfo="z",
    #         colorscale="Viridis",
    #         showscale=True,
    #     )
    #
    #     fig2 = ff.create_annotated_heatmap(
    #         z2,
    #         x=x,
    #         y=y,
    #         annotation_text=np.around(z2, decimals=4),
    #         hoverinfo="z",
    #         colorscale="Viridis",
    #         showscale=True,
    #     )
    #
    #     fig.layout.title = "Correlation Cramer Matrix"
    #     fig.layout.xaxis.title = "Category 1"
    #     fig.layout.yaxis.title = "Category 2"
    #     fig2.layout.title = "Correlation Tschuprow Matrix"
    #     fig2.layout.xaxis.title = "Category 1"
    #     fig2.layout.yaxis.title = "Category 2"
    #
    #     fig_html = plotly.io.to_html(fig, include_plotlyjs="cdn")
    #     fig2_html = plotly.io.to_html(fig2, include_plotlyjs="cdn")
    #
    #     # return tables with mean response plots
    #     return cor_tab_c, cor_tab_t, fig_html, fig2_html
    #
    # def cat_cont_matrix(data_table, col_names, var_dict, plot, mrplot):
    #     # corr table
    #
    #     cor_tab = pd.DataFrame(columns=["cat", "cont", "corr", "cat_url", "cont_url"])
    #
    #     # cat_cont col array
    #     cat_col = []
    #     cont_col = []
    #     for i in col_names:
    #         if var_dict[i] == "Categorical":
    #             cat_col.append(i)
    #         else:
    #             cont_col.append(i)
    #
    #     # n by m matrix
    #     zeros_array = np.zeros((len(cat_col), len(cont_col)))
    #
    #     for num, cat in enumerate(cat_col):
    #         for num2, cont in enumerate(cont_col):
    #             cat_r = fill_na(data_table[cat])
    #             cont_r = fill_na(data_table[cont])
    #             zeros_array[num][num2] = cat_cont_correlation_ratio(cat_r, cont_r)
    #             cor_tab.loc[len(cor_tab)] = [
    #                 cat,
    #                 cont,
    #                 cat_cont_correlation_ratio(cat_r, cont_r),
    #                 plot[cat],
    #                 mrplot[cont],
    #             ]
    #
    #     y = cat_col
    #     x = cont_col
    #     z = zeros_array
    #
    #     fig = ff.create_annotated_heatmap(
    #         z,
    #         x=x,
    #         y=y,
    #         annotation_text=np.around(z, decimals=4),
    #         hoverinfo="z",
    #         colorscale="Viridis",
    #         showscale=True,
    #     )
    #     fig.layout.title = "Continuous/Categorical Correlation Matrix"
    #     fig.layout.xaxis.title = "Category 1"
    #     fig.layout.yaxis.title = "Category 2"
    #     fig_html = plotly.io.to_html(fig, include_plotlyjs="cdn")
    #     return cor_tab, fig_html

    def cont_cont_matrix(data_table, col_names, var_dict, plot, mrplot):
        # corr table

        cor_tab = pd.DataFrame(
            columns=["cont_1", "cont_2", "corr", "cont_1_url", "cont_2_url"]
        )

        # cat_cont col array
        cont_col = []
        for i in col_names:
            if var_dict[i] == "Continuous":
                cont_col.append(i)

        # n by n matrix
        zeros_array = np.zeros((len(cont_col), len(cont_col)))

        for num, cont in enumerate(cont_col):
            for num2, cont2 in enumerate(cont_col):
                cont_1 = fill_na(data_table[cont])
                cont_2 = fill_na(data_table[cont2])
                zeros_array[num][num2] = stats.pearsonr(cont_1, cont_2)[0]
                cor_tab.loc[len(cor_tab)] = [
                    cont,
                    cont2,
                    stats.pearsonr(cont_1, cont_2)[0],
                    plot[cont],
                    mrplot[cont2],
                ]

        y = cont_col
        x = cont_col
        z = zeros_array

        fig = ff.create_annotated_heatmap(
            z,
            x=x,
            y=y,
            annotation_text=np.around(z, decimals=4),
            hoverinfo="z",
            colorscale="Viridis",
            showscale=True,
        )
        fig.layout.title = "Continuous/Continuous Correlation Matrix"
        fig.layout.xaxis.title = "Category 1"
        fig.layout.yaxis.title = "Category 2"
        fig_html = plotly.io.to_html(fig, include_plotlyjs="cdn")
        return cor_tab, fig_html

    # Brute force tables
    # def bf_cat_cat(data_table, response_name, corr_tab_c, corr_tab_t):
    #     bf_cat_cat_tab = pd.DataFrame(
    #         columns=[
    #             "cat_1",
    #             "cat_2",
    #             "diff_mean_resp_ranking",
    #             "diff_mean_resp_weighted_ranking",
    #             "cramer",
    #             "tschuprow",
    #             "abs_cramer",
    #             "abs_tschuprow",
    #             "link",
    #         ]
    #     )
    #
    #     bf_cat_cat_tab["cat_1"] = corr_tab_c["cat_1"]
    #     bf_cat_cat_tab["cat_2"] = corr_tab_c["cat_2"]
    #     bf_cat_cat_tab["cramer"] = corr_tab_c["corr"]
    #     bf_cat_cat_tab["tschuprow"] = corr_tab_t["corr"]
    #     bf_cat_cat_tab["abs_cramer"] = abs(corr_tab_c["corr"])
    #     bf_cat_cat_tab["abs_tschuprow"] = abs(corr_tab_t["corr"])
    #     bf_cat_cat_dmr = []
    #     bf_cat_cat_dmr_w = []
    #     plots = []
    #
    #     # if var_dict[response_name] == "Boolean":
    #     for cat, cat2 in zip(bf_cat_cat_tab["cat_1"], bf_cat_cat_tab["cat_2"]):
    #         zeros_array = np.zeros(
    #             (len(data_table[cat].unique()), len(data_table[cat2].unique()))
    #         )
    #         text = np.zeros(
    #             (len(data_table[cat].unique()), len(data_table[cat2].unique())),
    #             dtype="U20",
    #         )
    #         cat_type_row = data_table[cat].unique()
    #         cat_type_col = data_table[cat2].unique()
    #         cat_cat2_diff_mean = []
    #         cat_cat2_diff_mean_w = []
    #         bin_count = []
    #         for num3, i in enumerate(cat_type_row):
    #             for num4, w in enumerate(cat_type_col):
    #                 bin_avg_df = data_table[
    #                     (data_table[cat] == i) & (data_table[cat2] == w)
    #                 ]
    #                 bin_avg = np.mean(bin_avg_df[response_name])
    #                 zeros_array[num3][num4] = bin_avg
    #
    #                 if math.isnan(bin_avg):
    #                     text[num3][num4] = ""
    #                 else:
    #                     diff_mean = pow(
    #                         (bin_avg - np.mean(data_table[response_name])), 2
    #                     )
    #                     cat_cat2_diff_mean.append(diff_mean)
    #                     bin_pop = len(bin_avg_df) / len(data_table)
    #                     bin_count.append(len(bin_avg_df))
    #                     cat_cat2_diff_mean_w.append(diff_mean * bin_pop)
    #                     text_val = (
    #                         str(round(bin_avg, 6))
    #                         + ", "
    #                         + " pop("
    #                         + str(round(bin_pop, 2))
    #                         + ")"
    #                     )
    #                     text[num3][num4] = text_val
    #
    #                 zeros_array[num3][num4] = bin_avg
    #
    #         mr_fin = sum(cat_cat2_diff_mean) / (len(cat_cat2_diff_mean))
    #         mr_fin_w = sum(cat_cat2_diff_mean_w) / (len(cat_cat2_diff_mean_w))
    #         bf_cat_cat_dmr.append(mr_fin)
    #         bf_cat_cat_dmr_w.append(mr_fin_w)
    #
    #         # create plots
    #         x = cat_type_col.tolist()
    #         y = cat_type_row.tolist()
    #         z = zeros_array
    #         fig_bf = go.Figure(
    #             data=go.Heatmap(
    #                 z=z,
    #                 x=x,
    #                 y=y,
    #                 text=text,
    #                 texttemplate="%{text}",
    #             )
    #         )
    #         fig_bf.update_layout(
    #             title=f"{cat2} vs {cat}", xaxis_title=f"{cat2}", yaxis_title=f"{cat}"
    #         )
    #         fig_bf.write_html(file=f"Plots/{cat}_vs_{cat2}_plot.html")
    #
    #         plots.append(f"Plots/{cat}_vs_{cat2}_plot.html")
    #     # else:
    #     #     print("cont")
    #
    #     bf_cat_cat_tab["diff_mean_resp_ranking"] = bf_cat_cat_dmr
    #     bf_cat_cat_tab["diff_mean_resp_weighted_ranking"] = bf_cat_cat_dmr_w
    #     bf_cat_cat_tab["link"] = plots
    #
    #     return bf_cat_cat_tab
    #
    # def bf_cat_cont(data_table, response_name, corr_rat_t):
    #     bf_cat_cont_tab = pd.DataFrame(
    #         columns=[
    #             "cat",
    #             "cont",
    #             "diff_mean_resp_ranking",
    #             "diff_mean_resp_weighted_ranking",
    #             "corr_ratio",
    #             "abs_corr_ratio",
    #             "link",
    #         ]
    #     )
    #
    #     bf_cat_cont_tab["cat"] = corr_rat_t["cat"]
    #     bf_cat_cont_tab["cont"] = corr_rat_t["cont"]
    #     bf_cat_cont_tab["corr_ratio"] = abs(corr_rat_t["corr"])
    #     bf_cat_cont_tab["abs_corr_ratio"] = abs(corr_rat_t["corr"])
    #     bf_cat_cont_dmr = []
    #     bf_cat_cont_dmr_w = []
    #     plots = []
    #
    #     # if var_dict[response_name] == "Boolean":
    #     for cat, cont in zip(bf_cat_cont_tab["cat"], bf_cat_cont_tab["cont"]):
    #         cat_type_row = data_table[cat].unique()
    #         counts, bins = np.histogram(data_table[cont], bins=10)
    #         bins2 = 0.5 * (bins[:-1] + bins[1:])
    #         zeros_array = np.zeros((len(data_table[cat].unique()), len(bins2)))
    #         text = np.zeros((len(data_table[cat].unique()), len(bins2)), dtype="U20")
    #         cat_cont_diff_mean = []
    #         cat_cont_diff_mean_w = []
    #         for num, cat_i in enumerate(cat_type_row):
    #             for num2, cont_index in enumerate(bins2):
    #                 if cont_index == bins2[-1]:
    #                     break
    #                 # print(cat_i)
    #                 bin_avg_df = data_table[
    #                     (data_table[cat] == cat_i)
    #                     & (data_table[cont] >= cont_index)
    #                     & (data_table[cont] <= bins2[num2 + 1])
    #                 ]
    #                 bin_avg = np.mean(bin_avg_df[response_name])
    #                 zeros_array[num][num2] = bin_avg
    #
    #                 if math.isnan(bin_avg):
    #                     text[num][num2] = ""
    #                 else:
    #                     diff_mean = pow(
    #                         (bin_avg - np.mean(data_table[response_name])), 2
    #                     )
    #                     cat_cont_diff_mean.append(diff_mean)
    #                     bin_pop = len(bin_avg_df) / len(data_table)
    #                     cat_cont_diff_mean_w.append(diff_mean * bin_pop)
    #                     text_val = (
    #                         str(round(bin_avg, 6))
    #                         + ", "
    #                         + " pop("
    #                         + str(round(bin_pop, 2))
    #                         + ")"
    #                     )
    #                     text[num][num2] = text_val
    #
    #         mr_fin = sum(cat_cont_diff_mean) / (len(cat_cont_diff_mean))
    #         mr_fin_w = sum(cat_cont_diff_mean_w) / (len(cat_cont_diff_mean_w))
    #         bf_cat_cont_dmr.append(mr_fin)
    #         bf_cat_cont_dmr_w.append(mr_fin_w)
    #
    #         # create plots
    #         bins_lab = [str(z) for z in bins2]
    #         x = bins_lab
    #         y = cat_type_row.tolist()
    #         z = zeros_array
    #         fig_bf = go.Figure(
    #             data=go.Heatmap(
    #                 z=z,
    #                 x=x,
    #                 y=y,
    #                 text=text,
    #                 texttemplate="%{text}",
    #             )
    #         )
    #         fig_bf.update_layout(
    #             title=f"{cont} vs {cat}", xaxis_title=f"{cont}", yaxis_title=f"{cat}"
    #         )
    #         fig_bf.write_html(file=f"Plots/{cat}_vs_{cont}_plot.html")
    #
    #         plots.append(f"Plots/{cat}_vs_{cont}_plot.html")
    #
    #     # else:
    #     #     print("cont")
    #
    #     bf_cat_cont_tab["diff_mean_resp_ranking"] = bf_cat_cont_dmr
    #     bf_cat_cont_tab["diff_mean_resp_weighted_ranking"] = bf_cat_cont_dmr_w
    #     bf_cat_cont_tab["link"] = plots
    #
    #     return bf_cat_cont_tab

    def bf_cont_cont(data_table, response_name, corr_p_t):
        bf_cont_cont_tab = pd.DataFrame(
            columns=[
                "cont_1",
                "cont_2",
                "diff_mean_resp_ranking",
                "diff_mean_resp_weighted_ranking",
                "pearson",
                "abs_pearson",
                "link",
            ]
        )

        bf_cont_cont_tab["cont_1"] = corr_p_t["cont_1"]
        bf_cont_cont_tab["cont_2"] = corr_p_t["cont_2"]
        bf_cont_cont_tab["pearson"] = abs(corr_p_t["corr"])
        bf_cont_cont_tab["abs_pearson"] = abs(corr_p_t["corr"])
        bf_cont_cont_dmr = []
        bf_cont_cont_dmr_w = []
        plots = []

        # if var_dict[response_name] == "Boolean":
        for cont_1, cont_2 in zip(
            bf_cont_cont_tab["cont_1"], bf_cont_cont_tab["cont_2"]
        ):
            counts, bins = np.histogram(data_table[cont_1], bins=10)
            bins2 = 0.5 * (bins[:-1] + bins[1:])
            counts_2, bins_2 = np.histogram(data_table[cont_2], bins=10)
            bins_2_2 = 0.5 * (bins_2[:-1] + bins_2[1:])
            # print(bins2)
            zeros_array = np.zeros((len(bins2), len(bins_2_2)))
            text = np.zeros((len(bins2), len(bins_2_2)), dtype="U20")  # noqa:
            cont_cont_diff_mean = []
            cont_cont_diff_mean_w = []
            for num, i in enumerate(bins2):
                if i == bins2[-1]:
                    break
                # print(str(i)+", "+str(bins2[num+1]))
                for num2, j in enumerate(bins_2_2):
                    if j == bins_2_2[-1]:
                        break
                    # print(str(j)+", "+str(bins_2_2[num2+1]))
                    bin_avg_df = data_table[
                        (data_table[cont_1] >= i)
                        & (data_table[cont_1] <= bins2[num + 1])
                        & (data_table[cont_2] >= j)
                        & (data_table[cont_2] <= bins_2_2[num2 + 1])
                    ]
                    bin_avg = np.mean(bin_avg_df[response_name])
                    zeros_array[num][num2] = bin_avg

                    if math.isnan(bin_avg):
                        text[num][num2] = ""
                    else:
                        diff_mean = pow(
                            (bin_avg - np.mean(data_table[response_name])), 2
                        )
                        cont_cont_diff_mean.append(diff_mean)
                        bin_pop = len(bin_avg_df) / len(data_table)
                        cont_cont_diff_mean_w.append(diff_mean * bin_pop)
                        text_val = (
                            str(round(bin_avg, 6))
                            + ", "
                            + " pop("
                            + str(round(bin_pop, 2))
                            + ")"
                        )
                        text[num][num2] = text_val

            mr_fin = sum(cont_cont_diff_mean) / (len(cont_cont_diff_mean))
            mr_fin_w = sum(cont_cont_diff_mean_w) / (len(cont_cont_diff_mean_w))
            bf_cont_cont_dmr.append(mr_fin)
            bf_cont_cont_dmr_w.append(mr_fin_w)

            # create plots
            bins_lab1 = [str(z) for z in bins2]
            bins_lab2 = [str(z) for z in bins_2_2]

            x = bins_lab1
            y = bins_lab2
            z = zeros_array
            fig_bf = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    text=text,
                    texttemplate="%{text}",
                )
            )
            fig_bf.update_layout(
                title=f"{cont_1} vs {cont_2}",
                xaxis_title=f"{cont_1}",
                yaxis_title=f"{cont_2}",
            )
            fig_bf.write_html(file=f"Plots/{cont_1}_vs_{cont_2}_plot.html")

            plots.append(f"Plots/{cont_1}_vs_{cont_2}_plot.html")

        # else:
        #     print("cont")

        bf_cont_cont_tab["diff_mean_resp_ranking"] = bf_cont_cont_dmr
        bf_cont_cont_tab["diff_mean_resp_weighted_ranking"] = bf_cont_cont_dmr_w
        bf_cont_cont_tab["link"] = plots

        return bf_cont_cont_tab

    # Execution of HTML

    (
        p_value_t_value_table,
        mean_response_table,
        rf_imp_table,
        plots_table,
    ) = generate_html_m(final_table_f, predictors, cat_dic, response_var)

    if cat_dic[response_var] == "Boolean":
        result = pd.merge(plots_table, mean_response_table, on="Feature", how="left")
        result2 = pd.merge(result, rf_imp_table, on="Feature", how="left")
        result3 = pd.merge(result2, p_value_t_value_table, on="Feature", how="left")
        result_dic = result3.set_index("Feature").to_dict()
    else:
        result = pd.merge(plots_table, mean_response_table, on="Feature", how="left")
        result3 = pd.merge(result, p_value_t_value_table, on="Feature", how="left")
        result_dic = result3.set_index("Feature").to_dict()

    def make_clickable(url, names):
        return '<a href="{}">{}</a>'.format(url, names)

    result3["Plot"] = result3.apply(
        lambda x: make_clickable(x["Plot"], x["Feature"]), axis=1
    )

    result3["Mean of Response Plot"] = result3.apply(
        lambda x: make_clickable(x["Mean of Response Plot"], x["Feature"]), axis=1
    )

    # Make plot dictionary
    plot_dic = result_dic["Plot"]

    plot_mr_dic = result_dic["Mean of Response Plot"]

    # cat_c_tab, cat_t_tab, cat_c_fig, cat_t_fig = cat_cat_matrix(
    #     final_table_f, predictors, cat_dic, plot_dic, plot_mr_dic
    # )
    # cat_cont_tab, cat_cont_plot = cat_cont_matrix(
    #     final_table_f, predictors, cat_dic, plot_dic, plot_mr_dic
    # )

    cont_cont_tab, cont_cont_plot = cont_cont_matrix(
        final_table_f, predictors, cat_dic, plot_dic, plot_mr_dic
    )

    # bf_cat_cat_table = bf_cat_cat(final_table_f, response_var, cat_c_tab, cat_t_tab)
    #
    # bf_cat_cont_table = bf_cat_cont(final_table_f, response_var, cat_cont_tab)

    bf_cont_cont_table = bf_cont_cont(final_table_f, response_var, cont_cont_tab)

    # cat_c_tab["cat_1_url"] = cat_c_tab.apply(
    #     lambda x: make_clickable(x["cat_1_url"], x["cat_1"]), axis=1
    # )
    #
    # cat_c_tab["cat_2_url"] = cat_c_tab.apply(
    #     lambda x: make_clickable(x["cat_2_url"], x["cat_2"]), axis=1
    # )
    #
    # cat_t_tab["cat_1_url"] = cat_t_tab.apply(
    #     lambda x: make_clickable(x["cat_1_url"], x["cat_1"]), axis=1
    # )
    #
    # cat_t_tab["cat_2_url"] = cat_t_tab.apply(
    #     lambda x: make_clickable(x["cat_2_url"], x["cat_2"]), axis=1
    # )
    #
    # bf_cat_cat_table["link"] = bf_cat_cat_table.apply(
    #     lambda x: make_clickable(x["link"], "plot"), axis=1
    # )
    # cat_cont_tab["cat_url"] = cat_cont_tab.apply(
    #     lambda x: make_clickable(x["cat_url"], x["cat"]), axis=1
    # )
    # cat_cont_tab["cont_url"] = cat_cont_tab.apply(
    #     lambda x: make_clickable(x["cont_url"], x["cont"]), axis=1
    # )
    cont_cont_tab["cont_1_url"] = cont_cont_tab.apply(
        lambda x: make_clickable(x["cont_1_url"], x["cont_1"]), axis=1
    )
    cont_cont_tab["cont_2_url"] = cont_cont_tab.apply(
        lambda x: make_clickable(x["cont_2_url"], x["cont_2"]), axis=1
    )
    # bf_cat_cont_table["link"] = bf_cat_cont_table.apply(
    #     lambda x: make_clickable(x["link"], "plot"), axis=1
    # )
    bf_cont_cont_table["link"] = bf_cont_cont_table.apply(
        lambda x: make_clickable(x["link"], "plot"), axis=1
    )

    with open("HW_05.html", "w") as file:
        file.write("<h1>baseball</h1>")
        file.write(
            result3.to_html(
                header=True,
                index=False,
                render_links=True,  # make the links render
                escape=False,  # make sure the browser knows not to treat this as text only
            )
        )
        # file.write("<h2>Categorical/Categorical Correlations</h2>")
        # file.write(cat_c_fig)
        # file.write(cat_t_fig)
        # file.write("<h3>Correlation Cramer Table</h3>")
        # file.write(
        #     cat_c_tab[cat_c_tab["cat_1"] != cat_c_tab["cat_2"]]
        #     .sort_values(by="corr", ascending=False)
        #     .to_html(
        #         header=True,
        #         index=False,
        #         render_links=True,  # make the links render
        #         escape=False,  # make sure the browser knows not to treat this as text only
        #     )
        # )
        # file.write("<h3>Correlation Tschuprow Table</h3>")
        # file.write(
        #     cat_t_tab[cat_t_tab["cat_1"] != cat_t_tab["cat_2"]]
        #     .sort_values(by="corr", ascending=False)
        #     .to_html(
        #         header=True,
        #         index=False,
        #         render_links=True,  # make the links render
        #         escape=False,  # make sure the browser knows not to treat this as text only
        #     )
        # )
        # file.write("<h2>Continuous/Categorical Correlations</h2>")
        # file.write(cat_cont_plot)
        # file.write("<h3>Correlation Ratio Table</h3>")
        # file.write(
        #     cat_cont_tab.sort_values(by="corr", ascending=False).to_html(
        #         header=True,
        #         index=False,
        #         render_links=True,  # make the links render
        #         escape=False,  # make sure the browser knows not to treat this as text only
        #     )
        # )
        file.write("<h2>Continuous/Continuous Correlations</h2>")
        file.write(cont_cont_plot)
        file.write("<h3>Correlation Pearson's Table</h3>")
        file.write(
            cont_cont_tab[cont_cont_tab["cont_1"] != cont_cont_tab["cont_2"]]
            .sort_values(by="corr", ascending=False)
            .to_html(
                header=True,
                index=False,
                render_links=True,  # make the links render
                escape=False,  # make sure the browser knows not to treat this as text only
            )
        )
        # file.write("<h2>Categorical/Categorical Brute Force Table</h2>")
        # file.write(
        #     bf_cat_cat_table[bf_cat_cat_table["cat_1"] != bf_cat_cat_table["cat_2"]]
        #     .sort_values(by="diff_mean_resp_weighted_ranking", ascending=False)
        #     .to_html(
        #         header=True,
        #         index=False,
        #         render_links=True,  # make the links render
        #         escape=False,  # make sure the browser knows not to treat this as text only
        #     )
        # )
        # file.write("<h2>Categorical/Continuous - Brute Force</h2>")
        # file.write(
        #     bf_cat_cont_table.sort_values(
        #         by="diff_mean_resp_weighted_ranking", ascending=False
        #     ).to_html(
        #         header=True,
        #         index=False,
        #         render_links=True,  # make the links render
        #         escape=False,  # make sure the browser knows not to treat this as text only
        #     )
        # )
        file.write("<h2>Continuous/Continuous - Brute Force</h2>")
        file.write(
            bf_cont_cont_table[
                bf_cont_cont_table["cont_1"] != bf_cont_cont_table["cont_2"]
            ]
            .sort_values(by="diff_mean_resp_weighted_ranking", ascending=False)
            .to_html(
                header=True,
                index=False,
                render_links=True,  # make the links render
                escape=False,  # make sure the browser knows not to treat this as text only
            )
        )

    # Logistic Model

    X_train, X_test, y_train, y_test = train_test_split(
        final_table_f[predictors],
        final_table_f[response_var],  # noqa:
        test_size=0.3,
        random_state=42,
    )
    # Build the logistic regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the target variable for the test data
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    connection.close()


if __name__ == "__main__":
    sys.exit(main())
