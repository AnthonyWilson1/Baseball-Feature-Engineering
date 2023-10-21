# PythonProject

# Data:

I used baseball.sql which has tables of baseball data at the player and team level.
The tables I used specifically were team_batter_counts and pitcher_counts. Some columns are duplicated,
such as ground outs, etc. Although they have the same name, the data was different. This is one thing to note for the
analysis.

# Variables:

The variables I explored are the following:
plateApperance_home_team,
plateApperance_away_team,
toBase_home_team,
toBase_away_team,
Bunt_Ground_Out_home_team,
Bunt_Ground_Out_away_team,
Bunt_Pop_Out_home_team,
Bunt_Pop_Out_away_team,
Double_home_team,
Double_away_team,
Field_Error_home_team,
Field_Error_away_team,
Fly_Out_home_team,
Fly_Out_away_team,
Ground_Out_home_team,
Ground_Out_away_team,
Intent_Walk_home_team,
Intent_Walk_away_team,
Line_Out_home_team,
Line_Out_away_team,
Pop_Out_home_team,
Pop_Out_away_team,
Runner_Out_home_team,
Runner_Out_away_team,
Single_home_team,
Single_away_team,
Triple_home_team,
Triple_away_team,
Walk_home_team,
Walk_away_team,
Home_Run_home_team,
Home_Run_away_team,
RollingBattingAverage_home_team,
RollingBattingAverage_away_team,
DaysSinceLastPitch_diff,
home_team_endingInning,
away_team_endingInning,
home_team_Strikeout,
away_team_Strikeout,
home_team_pitchesThrown,
away_team_pitchesThrown,
HR_Hits_home_team (HR/Hits),
HR_Hits_away_team (HR/Hits)

# Final Model:

I used a Logistic Regression and split the training and testing data by the date of the games played. There is a 70/30
split of the testing and training data. The accuracy of the model is .63.

The variables in my final model with p-values below .05 are as follows:
plateApperance_home_team,
plateApperance_away_team,
toBase_home_team,
toBase_away_team,
Bunt_Pop_Out_home_team,
Double_home_team,
Double_away_team,
Field_Error_home_team,
Field_Error_away_team,
Fly_Out_home_team,
Fly_Out_away_team,
Ground_Out_home_team,
Ground_Out_away_team,
Line_Out_home_team,
Line_Out_away_team,
Pop_Out_home_team,
Pop_Out_away_team,
Runner_Out_home_team,
Single_home_team,
Single_away_team,
Walk_home_team,
Walk_away_team,
Home_Run_home_team,
RollingBattingAverage_home_team,
home_team_endingInning,
away_team_endingInning,
HR_Hits_away_team
