import sys

import numpy as np  # noqa:
import pandas as pd  # noqa:
from plotly import figure_factory as ff  # noqa:

from dataset_loader import TestDatasets


def main():
    dataset = TestDatasets()
    # Pull dataset
    all_dataset = dataset.get_test_data_set("titanic")

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

    dataset_table = all_dataset[0]

    check_response(dataset_table, response_var)

    # Check predictor variables to see if they ar cat or cont

    predictors = all_dataset[1]

    def check_predictors(data_table, col_names):
        for i in col_names:
            for j in data_table[i]:
                if type(j) == str:
                    cat_dic[i] = "Discrete"
                else:
                    cat_dic[i] = "Continuous"
                break

    check_predictors(dataset_table, predictors)

    # Generate Plots

    def generate_plots(data_table, col_names, bin_n, var_dict, response_name):  # noqa:
        if var_dict[response_name] == "Boolean":
            for i in col_names:
                # continuous predictor vs. boolean response
                if cat_dic[i] == "Continuous":
                    group_labels = ["Response = 0", "Response = 1"]  # noqa:
                    pred_0 = data_table[response_name] == 0
                    pred_1 = data_table[response_name] == 1

                    bin_1 = data_table[pred_0]
                    bin_2 = data_table[pred_1]  # noqa:

                    print(bin_1)

                    # # Create distribution plot with custom bin_size
                    # fig_1 = ff.create_distplot(data_table[i], group_labels, bin_size=0.2)
                    # fig_1.update_layout(
                    #     title="Continuous Predictor by Categorical Response",
                    #     xaxis_title="Predictor",
                    #     yaxis_title="Distribution",
                    # )
                    # fig_1.show()

    generate_plots(dataset_table, predictors, 10, cat_dic, response_var)


if __name__ == "__main__":
    sys.exit(main())
