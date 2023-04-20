import sys

import pandas as pd
import sqlalchemy
from sqlalchemy import text


def main():
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

    # response is home team wins (0,1)
    # each row is going to be a game a team plays
    # have two teams and the response is that the home team wins (0,1)
    # team_results then filter for home_games
    # each row will be one game with team level stats before that point

    # Feature 1 avg home_runs_home_team
    # Feature 2 avg home_runs_away_team
    # Feature 3 avg strike_outs
    # Feature 4 away_streak
    # Feature 5 away_runs
    # Feature 6 away_errors
    # Feature 7 avg hits
    # Feature 8 batting average
    # Feature 9 avg home_error
    # Feature 10 starting pitcher (0,1) and the pitcher ID with their stats
    # Feature 11 odds home team
    # Feature 12 pitcher walk
    # Feature 13 opposing pitcher (0,1) and the pitcher ID with their stats

    # Filter Response
    query_response = text("SELECT * FROM  team_results WHERE home_away = 'H'")
    df_response = pd.read_sql_query(query_response, connection)
    df_response_final = df_response[  # noqa:
        [
            "team_results_id",
            "game_id",
            "team_id",
            "opponent_id",
            "home_away",
            "win_lose",
            "home_streak",
        ]
    ]

    # Filter First Feature 100 day rolling average home runs
    # Join Game and Team Batter Counts in a temp table to get the dates,
    # then join the table with itself so home_team and away_team are on the same row
    # get the 100 day rolling average by merging with itself
    # Home_Run, team_id, opponent_team_id, homeTeam, awayTeam

    temp_team_counts_query = text(
        """CREATE TEMPORARY TABLE team_batting_counts_dates AS
        (SELECT tc.game_id AS game_id_tc, gam.local_date AS game_date,
        tc.team_id AS team_tc, tc.Home_Run AS Home_Run_tc,
        tc.opponent_team_id AS opponent_team_id_tc,
        tc.homeTeam AS homeTeam_tc, tc.awayTeam as awayTeam_tc
        FROM team_batting_counts tc
        JOIN game gam ON tc.game_id = gam.game_id)"""
    )

    # temp_home_away_same = text('''CREATE TEMPORARY TABLE team_batting_counts_away_same AS
    # (SELECT ha1.game_id AS game_id_ha1, ha2.game_id AS game_id_ha2)''')

    connection.execute(temp_team_counts_query)
    pull_temp_query = text("SELECT * FROM team_batting_counts_dates")
    result = pd.read_sql_query(pull_temp_query, connection)

    df_team_counts = pd.read_sql_query(  # noqa:
        text("""SELECT * FROM team_batting_counts"""), connection
    )  # noqa:

    print(result.head())

    # print(df_team_counts.shape)

    connection.close()


if __name__ == "__main__":
    sys.exit(main())
