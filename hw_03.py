import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession


# Rolling Average Transformer
class RollingAverageTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):  # noqa:
        super(RollingAverageTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setparams(**kwargs)
        return

    @keyword_only
    def setparams(self, inputCols=None, outputCol=None):  # noqa:
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):  # noqa:
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        spark = (
            SparkSession.builder.master("local[*]")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.debug.maxToStringFields", 1000)
            .getOrCreate()
        )

        rolling_avg = spark.sql(
            f"""
            SELECT
            table_one.{input_cols[0]} as game_id_f
            ,table_one.{input_cols[1]} as game_date_start
            ,table_one.{input_cols[2]} as batter_F
            ,SUM(table_two.{input_cols[3]}) as Hits_two
            ,SUM(table_two.{input_cols[4]}) as atBat_two
            ,IF (SUM(table_two.{input_cols[4]})>0, SUM(table_two.{input_cols[3]})
            / SUM(table_two.{input_cols[4]}), 0) AS {output_col}
                FROM intermediate_df AS table_one
                    LEFT JOIN intermediate_df AS table_two
                        ON table_one.{input_cols[2]} = table_two.{input_cols[2]}
                        AND table_two.{input_cols[1]} >= table_one.{input_cols[1]} - INTERVAL '100' DAY
                        AND table_two.{input_cols[1]} < table_one.{input_cols[1]}
            GROUP BY batter_F, game_date_start, game_id_f
            """
        )
        return rolling_avg


def main():
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.debug.maxToStringFields", 1000)
        .getOrCreate()
    )

    jdbc_driver = "org.mariadb.jdbc.Driver"

    # batter_counts table
    # https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database (Source)
    batter_counts = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.batter_counts")
        .option("user", "test")
        .option("password", "test")  # pragma: allowlist secret
        .option("driver", jdbc_driver)
        .load()
    )

    # game table
    game = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.game")
        .option("user", "test")
        .option("password", "test")  # pragma: allowlist secret
        .option("driver", jdbc_driver)
        .load()
    )

    # create a temp table for batter_counts and have it persist at disk space
    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.DISK_ONLY)

    # create a temp table for game and have it persist at disk space
    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.DISK_ONLY)

    # create intermediate table
    intermediate_df = spark.sql(
        """
            SELECT bc.game_id as game_id_bc, big.game_id as game_id_big,
            bc.updatedDate AS updatedDate_bc, big.local_date AS game_date,
            bc.batter AS batter_bc,
            bc.Hit AS Hit_bc, bc.atBat AS atBat_bc
                FROM batter_counts bc
                    JOIN game big
                        ON bc.game_id = big.game_id
            """
    )

    # create a temp table for intermediate_df and have it persist at disk space
    intermediate_df.createOrReplaceTempView("intermediate_df ")
    intermediate_df.persist(StorageLevel.DISK_ONLY)

    # use transformer to return the 100 day rolling average table
    rolling_average_t = RollingAverageTransform(
        inputCols=["game_id_bc", "game_date", "batter_bc", "Hit_bc", "atBat_bc"],
        outputCol="rolling_batting_average",
    )

    rolling_average = rolling_average_t.transform(intermediate_df)
    rolling_average.show(100)


if __name__ == "__main__":
    sys.exit(main())
