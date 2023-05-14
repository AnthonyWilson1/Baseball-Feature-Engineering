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

    # index_game_id = text("CREATE INDEX idx_game_id ON team_batting_counts (game_id)")

    # connection.execute(index_game_id)

    # index_game_id = text("CREATE INDEX idx_team_id ON team_batting_counts (team_id)")

    # connection.execute(index_game_id)

    # index_game_id_p = text("CREATE INDEX idx_game_id_p ON pitcher_counts (game_id)")

    # connection.execute(index_game_id_p)

    # index_game_id_p = text("CREATE INDEX idx_team_id_p ON pitcher_counts (team_id)")

    # connection.execute(index_game_id_p)

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
        (SELECT tc.*, gam.local_date AS game_date
        FROM team_batting_counts tc
        JOIN game gam ON tc.game_id = gam.game_id)"""
    )

    temp_home_away_same = text(
        """CREATE TEMPORARY TABLE team_batting_counts_away_same AS
        (SELECT orig.game_id AS game_id_orig, copy.game_id AS game_id_copy,
        orig.game_date AS game_date_orig, orig.team_id AS home_team,
        copy.team_id AS away_team, orig.Home_Run AS Home_Run_home_team,
        copy.Home_Run AS Home_Run_away_team,
        orig.atBat AS atBat_home_team, copy.atBat AS atBat_away_team,
        orig.Hit AS Hit_home_team, copy.Hit AS Hit_away_team,
        orig.plateApperance AS plateApperance_home_team,
        copy.plateApperance AS plateApperance_away_team,
        orig.caughtStealing2B AS caughtStealing2B_home_team,
        copy.caughtStealing2B AS caughtStealing2B_away_team,
        orig.caughtStealing3B AS caughtStealing3B_home_team,
        copy.caughtStealing3B AS caughtStealing3B_away_team,
        orig.caughtStealingHome AS caughtStealingHome_home_team,
        copy.caughtStealingHome AS caughtStealingHome_away_team,
        orig.stolenBase2B AS stolenBase2B_home_team,
        copy.stolenBase2B AS stolenBase2B_away_team,
        orig.stolenBase3B AS stolenBase3B_home_team,
        copy.stolenBase3B AS stolenBase3B_away_team,
        orig.stolenBaseHome AS stolenBaseHome_home_team,
        copy.stolenBaseHome AS stolenBaseHome_away_team,
        orig.toBase AS toBase_home_team,
        copy.toBase AS toBase_away_team,
        orig.Batter_Interference AS Batter_Interference_home_team,
        copy.Batter_Interference AS Batter_Interference_away_team,
        orig.Bunt_Ground_Out AS Bunt_Ground_Out_home_team,
        copy.Bunt_Ground_Out AS Bunt_Ground_Out_away_team,
        orig.Bunt_Groundout AS Bunt_Groundout_home_team,
        copy.Bunt_Groundout AS Bunt_Groundout_away_team,
        orig.Bunt_Pop_Out AS Bunt_Pop_Out_home_team,
        copy.Bunt_Pop_Out AS Bunt_Pop_Out_away_team,
        orig.Catcher_Interference AS Catcher_Interference_home_team,
        copy.Catcher_Interference AS Catcher_Interference_away_team,
        orig.Double AS Double_home_team,
        copy.Double AS Double_away_team,
        orig.Double_Play AS Double_Play_home_team,
        copy.Double_Play AS Double_Play_away_team,
        orig.Fan_interference AS Fan_interference_home_team,
        copy.Fan_interference AS Fan_interference_away_team,
        orig.Field_Error AS Field_Error_home_team,
        copy.Field_Error AS Field_Error_away_team,
        orig.Fielders_Choice AS Fielders_Choice_home_team,
        copy.Fielders_Choice AS Fielders_Choice_away_team,
        orig.Fielders_Choice_out AS Fielders_Choice_out_home_team,
        copy.Fielders_Choice_out AS Fielders_Choice_out_away_team,
        orig.Fly_Out AS Fly_Out_home_team,
        copy.Fly_Out AS Fly_Out_away_team,
        orig.Flyout AS Flyout_home_team,
        copy.Flyout AS Flyout_away_team,
        orig.Force_Out AS Force_Out_home_team,
        copy.Force_Out AS Force_Out_away_team,
        orig.Forceout AS Forceout_home_team,
        copy.Forceout AS Forceout_away_team,
        orig.Ground_Out AS Ground_Out_home_team,
        copy.Ground_Out AS Ground_Out_away_team,
        orig.Grounded_Into_DP AS Grounded_Into_DP_home_team,
        copy.Grounded_Into_DP AS Grounded_Into_DP_away_team,
        orig.Groundout AS Groundout_home_team,
        copy.Groundout AS Groundout_away_team,
        orig.Hit_By_Pitch AS Hit_By_Pitch_home_team,
        copy.Hit_By_Pitch AS Hit_By_Pitch_away_team,
        orig.Intent_Walk AS Intent_Walk_home_team,
        copy.Intent_Walk AS Intent_Walk_away_team,
        orig.Line_Out AS Line_Out_home_team,
        copy.Line_Out AS Line_Out_away_team,
        orig.Lineout AS Lineout_home_team,
        copy.Lineout AS Lineout_away_team,
        orig.Pop_Out AS Pop_Out_home_team,
        copy.Pop_Out AS Pop_Out_away_team,
        orig.Runner_Out AS Runner_Out_home_team,
        copy.Runner_Out AS Runner_Out_away_team,
        orig.Sac_Bunt AS Sac_Bunt_home_team,
        copy.Sac_Bunt AS Sac_Bunt_away_team,
        orig.Sac_Fly AS Sac_Fly_home_team,
        copy.Sac_Fly AS Sac_Fly_away_team,
        orig.Sac_Fly_DP AS Sac_Fly_DP_home_team,
        copy.Sac_Fly_DP AS Sac_Fly_DP_away_team,
        orig.Sacrifice_Bunt_DP AS Sacrifice_Bunt_DP_home_team,
        copy.Sacrifice_Bunt_DP AS Sacrifice_Bunt_DP_away_team,
        orig.Single AS Single_home_team,
        copy.Single AS Single_away_team,
        orig.Triple AS Triple_home_team,
        copy.Triple AS Triple_away_team,
        copy.Triple_Play AS Triple_Play_away_team,
        orig.Triple_Play AS Triple_Play_home_team,
        orig.Walk AS Walk_home_team,
        copy.Walk AS Walk_away_team
        FROM team_batting_counts_dates orig
        JOIN team_batting_counts_dates copy ON orig.game_id = copy.game_id
        WHERE orig.team_id != copy.team_id
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
        """ SELECT  # noqa:
        table_one.*
        ,COUNT(table_two.home_team) as counts
        ,IF (SUM(table_two.atBat_home_team)>0, SUM(table_two.Hit_home_team) / SUM(table_two.atBat_home_team),
         0) AS RollingBattingAverage_home_team
        ,IF (SUM(table_two.atBat_away_team)>0, SUM(table_two.Hit_away_team) / SUM(table_two.atBat_away_team),
         0) AS RollingBattingAverage_away_team
        FROM team_batting_counts_no_dup AS table_one
        LEFT JOIN team_batting_counts_no_dup AS table_two
        ON table_one.home_team = table_two.home_team
        AND table_two.game_date_orig >= DATE_SUB(table_one.game_date_orig,INTERVAL 100 DAY)
        AND table_two.game_date_orig < table_one.game_date_orig
        GROUP BY table_one.home_team, table_one.game_date_orig, table_one.game_id_orig
        """
    )

    final = pd.read_sql_query(temp_rolling_atbat, connection)
    print(final.head())

    # Pitching Stats

    sp_temp = text(
        """CREATE TEMPORARY TABLE starting_pitcher_counts AS
        SELECT * FROM pitcher_counts WHERE startingPitcher = 1"""
    )

    sp_temp_date = text(
        """CREATE TEMPORARY TABLE starting_pitcher_counts_date AS
        (SELECT tc.*, gam.local_date AS game_date
        FROM starting_pitcher_counts tc
        JOIN game gam ON tc.game_id = gam.game_id)"""
    )

    temp_home_away_same_pitcher = text(
        """CREATE TEMPORARY TABLE starting_pitcher_counts_away_same AS
        (SELECT orig.game_id AS game_id_orig, copy.game_id AS game_id_copy,
        orig.game_date AS game_date_orig, orig.team_id AS home_team,
        copy.team_id AS away_team, orig.pitchesThrown AS home_team_pitches_thrown,
        copy.pitchesThrown AS away_team_pitches_thrown,
        orig.DaysSinceLastPitch AS home_team_DaysSinceLastPitch,
        copy.DaysSinceLastPitch AS away_team_DaysSinceLastPitch,
        orig.pitchesThrown AS home_team_pitchesThrown,
        copy.pitchesThrown AS away_team_pitchesThrown,
        orig.Strikeout AS home_team_Strikeout,
        copy.Strikeout AS away_team_Strikeout,
        orig.endingInning AS home_team_endingInning,
        orig.endingInning AS away_team_endingInning,
        orig.Home_Run AS home_team_Home_Run,
        orig.pitcher AS home_team_pitcher,
        copy.pitcher AS away_team_pitcher
        FROM starting_pitcher_counts_date orig
        JOIN starting_pitcher_counts_date AS copy ON orig.game_id = copy.game_id
        WHERE orig.team_id != copy.team_id
        )"""
    )

    connection.execute(sp_temp)
    connection.execute(sp_temp_date)
    connection.execute(temp_home_away_same_pitcher)

    df_pitcher = pd.read_sql_query(  # noqa:
        text(
            """SELECT game_id_orig, home_team_DaysSinceLastPitch, home_team_endingInning,home_team_Home_Run,
            home_team_Strikeout, away_team_Strikeout,away_team_endingInning, home_team_pitchesThrown,
            away_team_pitchesThrown, away_team_DaysSinceLastPitch FROM starting_pitcher_counts_away_same"""
        ),
        connection,
    )  # noqa:

    features = pd.merge(final, df_pitcher, on="game_id_orig", how="outer")

    features_f = features.dropna()

    final_table = pd.merge(features_f, df_response, on="game_id_orig", how="outer")

    final_table_f = final_table.dropna()

    final_table_f["win_lose"] = final_table_f["win_lose"].replace({"W": 1, "L": 0})
    final_table_f["DaysSinceLastPitch_diff"] = abs(
        final_table_f["home_team_DaysSinceLastPitch"]
        - final_table_f["away_team_DaysSinceLastPitch"]
    )
    final_table_f["HR_Hits_home_team"] = np.nan_to_num(
        abs(final_table_f["Home_Run_home_team"] / final_table_f["Hit_home_team"])
    )
    final_table_f["HR_Hits_away_team"] = np.nan_to_num(
        abs(final_table_f["Home_Run_away_team"] / final_table_f["Hit_away_team"])
    )

    # print(final_table_f.head())

    predictors = [
        "plateApperance_home_team",
        "plateApperance_away_team",
        "toBase_home_team",
        "toBase_away_team",
        "Bunt_Ground_Out_home_team",
        "Bunt_Ground_Out_away_team",
        "Bunt_Pop_Out_home_team",
        "Bunt_Pop_Out_away_team",
        "Double_home_team",
        "Double_away_team",
        "Field_Error_home_team",
        "Field_Error_away_team",
        "Fly_Out_home_team",
        "Fly_Out_away_team",
        "Ground_Out_home_team",
        "Ground_Out_away_team",
        "Intent_Walk_home_team",
        "Intent_Walk_away_team",
        "Line_Out_home_team",
        "Line_Out_away_team",
        "Pop_Out_home_team",
        "Pop_Out_away_team",
        "Runner_Out_home_team",
        "Runner_Out_away_team",
        "Single_home_team",
        "Single_away_team",
        "Triple_home_team",
        "Triple_away_team",
        "Walk_home_team",
        "Walk_away_team",
        "Home_Run_home_team",
        "Home_Run_away_team",
        "RollingBattingAverage_home_team",
        "RollingBattingAverage_away_team",
        "DaysSinceLastPitch_diff",
        "home_team_endingInning",
        "away_team_endingInning",
        "home_team_Strikeout",
        "away_team_Strikeout",
        "home_team_pitchesThrown",
        "away_team_pitchesThrown",
        "HR_Hits_home_team",
        "HR_Hits_away_team",
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

    cont_cont_tab, cont_cont_plot = cont_cont_matrix(
        final_table_f, predictors, cat_dic, plot_dic, plot_mr_dic
    )

    bf_cont_cont_table = bf_cont_cont(final_table_f, response_var, cont_cont_tab)

    cont_cont_tab["cont_1_url"] = cont_cont_tab.apply(
        lambda x: make_clickable(x["cont_1_url"], x["cont_1"]), axis=1
    )
    cont_cont_tab["cont_2_url"] = cont_cont_tab.apply(
        lambda x: make_clickable(x["cont_2_url"], x["cont_2"]), axis=1
    )

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

    predictors_f = [
        "plateApperance_home_team",
        "plateApperance_away_team",
        "toBase_home_team",
        "toBase_away_team",
        "Bunt_Pop_Out_home_team",
        "Double_home_team",
        "Double_away_team",
        "Field_Error_home_team",
        "Field_Error_away_team",
        "Fly_Out_home_team",
        "Fly_Out_away_team",
        "Ground_Out_home_team",
        "Ground_Out_away_team",
        "Line_Out_away_team",
        "Pop_Out_home_team",
        "Pop_Out_away_team",
        "Runner_Out_home_team",
        "Single_home_team",
        "Single_away_team",
        "Walk_home_team",
        "Walk_away_team",
        "Home_Run_home_team",
        "RollingBattingAverage_home_team",
        "home_team_endingInning",
        "away_team_endingInning",
        "HR_Hits_away_team",
    ]

    # Split the data based on game_date_orig column
    X_train, X_test, y_train, y_test = train_test_split(  # noqa:
        final_table_f[predictors_f],
        final_table_f[response_var],
        test_size=0.3,
        random_state=42,
        stratify=final_table_f["game_date_orig"].dt.date,
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
