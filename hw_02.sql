USE baseball;

CREATE TABLE Batters_Batting_Average
AS
SELECT batter, COALESCE(SUM(Hit) / SUM(atBat)) AS BattingAverage
FROM batter_counts
GROUP BY batter
;

CREATE TABLE Annual_Batting_Average
AS
SELECT EXTRACT(YEAR FROM updatedDate) AS ba_year, COALESCE(SUM(Hit) / SUM(atBat)) AS BattingAverage, batter
FROM batter_counts
GROUP BY batter, ba_year
;

CREATE TABLE Rolling_Batting_Average
AS
SELECT tbl1.batter AS batter_a, tbl2.batter AS batter_b, tbl1.updatedDate AS updatedDate_a, tbl2.updatedDate AS updatedDate_b, COALESCE(SUM(tbl1.Hit) / SUM(tbl2.atBat)) AS RollingBattingAverage
FROM batter_counts tbl1
    LEFT JOIN batter_counts tbl2
        ON tbl2.updatedDate >= tbl1.updatedDate - INTERVAL '100' DAY
            AND tbl2.updatedDate < tbl1.updatedDate
GROUP BY batter_a, updatedDate_a
;
