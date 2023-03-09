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
    def __init__(self, input_cols=None, output_col=None):
        super(RollingAverageTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setparams(**kwargs)
        return

    @keyword_only
    def setparams(self, input_cols=None, output_col=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()
        print(input_cols)
        print(output_col)
        return dataset


def main():
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
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
        .option("dbtable", "baseball.batter_counts")
        .option("user", "test")
        .option("password", "test")  # pragma: allowlist secret
        .option("driver", jdbc_driver)
        .load()
    )

    # create a temp table for batter_counts and have it persist at disk space
    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.DISK_ONLY)

    # create a game table for batter_counts and have it persist at disk space
    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.DISK_ONLY)


if __name__ == "__main__":
    sys.exit(main())
