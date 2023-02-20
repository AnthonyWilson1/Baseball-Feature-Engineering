import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mean_response_plot_up(df, pred, class_name, bin_n):
    # mean_response_plot_up(data, "sepal_len", "is_setosa", Iris-setosa, 10)

    # Add response columns to iris dataframe
    df[f"is_{class_name}"] = (df["class"] == class_name) * 1

    # Plot Mean Response Plot
    fig = go.Figure()
    counts, bins = np.histogram(df[pred], bins=bin_n)
    bins2 = 0.5 * (bins[:-1] + bins[1:])
    trace0 = go.Bar(x=bins2, y=counts, name=pred, yaxis="y2", opacity=0.5)

    class_avg = [df[f"is_{class_name}"].value_counts(normalize=True)[1]] * bin_n

    trace1 = go.Scatter(x=bins2, y=class_avg, name=rf"$\mu_{{{pred[:3]}}}$")

    bin_avg = []

    j = 0
    k = 1

    for i in range(0, bin_n):

        df_b = df[(df[pred] >= bins[j]) & (df[pred] < bins[k])]

        try:
            bin_avg.append(df_b[f"is_{class_name}"].value_counts(normalize=True)[1])
        except KeyError:
            bin_avg.append(0)

        j += 1
        k += 1

    trace2 = go.Scatter(x=bins2, y=bin_avg, name=rf"$\mu_{{i}}-\mu_{{{pred[:3]}}}$")

    fig.update_layout(
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Response"),
            side="left",
            range=[0, 1.1],
        ),
        yaxis2=dict(
            title=dict(text=pred),
            side="right",
            range=[0, max(counts) + 1],
            overlaying="y",
            tickmode="sync",
        ),
    )
    # Add figure title
    fig.update_layout(title_text=f"is_{class_name}")

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bin")

    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.show()


# Function for Summary Statistics
def summary_stat(df):
    column_name = df.columns.values.tolist()
    for i in column_name[:-1]:
        print("Average: " + str(i) + " " + str(np.average(df[i])))
        print("Min: " + str(i) + " " + str(np.min(df[i])))
        print("Max: " + str(i) + " " + str(np.max(df[i])))
        print(".25 Quartile: " + str(i) + " " + str(np.quantile(df[i], 0.25)))
        print(".75 Quartile: " + str(i) + " " + str(np.quantile(df[i], 0.75)))


# Function for Plots
def plots(df, category):
    # Avg bar plot

    class_values = df[category].unique()
    columns = df.columns.values.tolist()
    full_avg = []
    dfs = {}

    # https://stackoverflow.com/questions/53887292/pandas-dataframe-filter-and-for-loop
    for f_class, df_class in df.groupby(category):
        dfs[f_class] = df_class

    for i in class_values:
        aver = {"attributes": columns[:-1], "average": [], "plant": i}
        for j in columns[:-1]:
            aver["average"].append(np.average(dfs[i][j]))
        df2 = pd.DataFrame(aver)
        full_avg.append(df2)

    df_fin_avg = pd.concat(full_avg)

    fig1 = px.bar(df_fin_avg, x="attributes", y="average", color="plant")
    fig1.show()

    # Scatter plot for sepal width vs sepal length
    fig2 = px.scatter(
        df,
        x="sepal_wid",
        y="sepal_len",
        color="class",
        size="petal_len",
        hover_data=["petal_wid"],
    )
    fig2.show()

    # Violin plot
    fig4 = px.violin(
        df,
        y="petal_len",
        x="petal_wid",
        color="class",
        box=True,
        points="all",
        hover_data=df.columns,
    )
    fig4.show()

    # Histogram
    fig5 = px.histogram(df, x="petal_len", color="class", nbins=10)
    fig5.show()


def ml_models(df, response):

    columns = df.columns.values.tolist()

    # Add response columns to iris dataframe
    df[f"is_{response}"] = (df["class"] == response) * 1

    # ML Model for response class
    big_x_orig = df[columns[:-1]].values
    y_response = df[f"is_{response}"].values

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(big_x_orig)
    x_orig = scaler.transform(big_x_orig)

    # Fit the features to a random forrest
    random_forest_response = RandomForestClassifier(random_state=1234)
    random_forest_response.fit(x_orig, y_response)
    test_data = df[columns[0:4]]
    x_orig_test = test_data.values
    trans_test_data = scaler.transform(x_orig_test)
    prediction_response = random_forest_response.predict(trans_test_data)
    probability_response = random_forest_response.predict_proba(trans_test_data)
    print(f"Probability: {probability_response}")
    print(f"Predictions: {prediction_response}")

    # K Nearest Neighbor
    kn_response = KNeighborsClassifier(n_neighbors=3)
    kn_response.fit(x_orig, y_response)
    kn_response_pred = kn_response.predict(trans_test_data)
    print(f"Predictions_neigh_response: {kn_response_pred}")

    # Fit the features to bayes classifier
    clf_response = GaussianNB()
    clf_response.fit(x_orig, y_response)
    clf_response_pred = clf_response.predict(trans_test_data)
    print(f"Predictions_clf_response: {clf_response_pred}")


def ml_pipline(df, y):

    x = df[df.columns[:4]]

    df[f"is_{y}"] = (df["class"] == y) * 1

    y_response = df[f"is_{y}"].values

    # As pipeline using random forrest
    pipeline_set = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    pipeline_set.fit(x, y_response)

    probability_set_pip = pipeline_set.predict_proba(x)
    prediction_set_pip = pipeline_set.predict(x)
    print(f"Probability_pip: {probability_set_pip}")
    print(f"Predictions_pip: {prediction_set_pip}")


def main():
    # load in csv file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    data = pd.read_csv(url, header=None)

    columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]

    data.columns = columns

    summary_stat(data)

    plots(data, "class")

    ml_models(data, "Iris-setosa")

    ml_pipline(data, "Iris-setosa")

    # Mean Response Plot Function
    classes = data["class"].unique()
    for i in classes:
        for j in data[data.columns[:-1]]:
            mean_response_plot_up(data, j, i, 10)


if __name__ == "__main__":
    sys.exit(main())
