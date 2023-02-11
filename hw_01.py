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


def mean_response_plot(pred, resp_var):
    # load in csv file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    data = pd.read_csv(url, header=None)

    # Format Data
    columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    data.columns = columns
    data["is_setosa"] = (data["class"] == "Iris-setosa") * 1
    data["is_versicolor"] = (data["class"] == "Iris-versicolor") * 1
    data["is_virginica"] = (data["class"] == "Iris-virginica") * 1

    # Plot Mean Response Plot
    fig = go.Figure()
    counts, bins = np.histogram(data[pred], bins=10)
    bins2 = 0.5 * (bins[:-1] + bins[1:])
    trace0 = go.Bar(x=bins2, y=counts, name="Predictor", yaxis="y2", opacity=0.5)

    class_avg = [data[resp_var].value_counts(normalize=True)[1]] * 10

    trace1 = go.Scatter(x=bins2, y=class_avg, name="Average")

    df_1b = data[(data[pred] >= bins[0]) & (data[pred] < bins[1])]
    df_2b = data[(data[pred] >= bins[1]) & (data[pred] < bins[2])]
    df_3b = data[(data[pred] >= bins[2]) & (data[pred] < bins[3])]
    df_4b = data[(data[pred] >= bins[3]) & (data[pred] < bins[4])]
    df_5b = data[(data[pred] >= bins[4]) & (data[pred] < bins[5])]
    df_6b = data[(data[pred] >= bins[5]) & (data[pred] < bins[6])]
    df_7b = data[(data[pred] >= bins[6]) & (data[pred] < bins[7])]
    df_8b = data[(data[pred] >= bins[7]) & (data[pred] < bins[8])]
    df_9b = data[(data[pred] >= bins[8]) & (data[pred] < bins[9])]
    df_10b = data[(data[pred] >= bins[9]) & (data[pred] < bins[10])]

    df = []

    try:
        df.append(df_1b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_2b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_3b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_4b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_5b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_6b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_7b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_8b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_9b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    try:
        df.append(df_10b[resp_var].value_counts(normalize=True)[1])
    except KeyError:
        df.append(0)

    trace2 = go.Scatter(x=bins2, y=df, name="Predictor Average")

    fig.update_layout(
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Average"),
            side="left",
            range=[0, 1.1],
        ),
        yaxis2=dict(
            title=dict(text="Predictor"),
            side="right",
            range=[0, max(counts) + 1],
            overlaying="y",
            tickmode="sync",
        ),
    )

    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.show()
    return


def main():
    # load in csv file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    data = pd.read_csv(url, header=None)

    columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]

    # average all classes
    average_sepal_len = np.average(data[0])
    print(average_sepal_len)

    # min all classes
    min_sepal_len = np.min(data[0])
    print(min_sepal_len)

    # max all classes
    max_sepal_len = np.max(data[0])
    print(max_sepal_len)

    # first-quartile all classes
    fir_q_sepal_len = np.quantile(data[0], 0.25)
    print(fir_q_sepal_len)

    # third-quartile all classes
    third_q_sepal_len = np.quantile(data[0], 0.75)
    print(third_q_sepal_len)

    # plots of different classes
    setosa = data[data[4] == "Iris-setosa"]
    versicolor = data[data[4] == "Iris-versicolor"]
    virginica = data[data[4] == "Iris-virginica"]

    # Bar plot comparing the averages for the sepal and petal for each class of flower

    flower_classes = ["sep_len", "sep_wid", "ped_len", "ped_wid"]

    avg_setosa = {
        "attributes": flower_classes,
        "average": [
            np.average(setosa[0]),
            np.average(setosa[1]),
            np.average(setosa[2]),
            np.average(setosa[3]),
        ],
        "plant": "setosa",
    }

    avg_versicolor = {
        "attributes": flower_classes,
        "average": [
            np.average(versicolor[0]),
            np.average(versicolor[1]),
            np.average(versicolor[2]),
            np.average(versicolor[3]),
        ],
        "plant": "versicolor",
    }

    avg_virginica = {
        "attributes": flower_classes,
        "average": [
            np.average(virginica[0]),
            np.average(virginica[1]),
            np.average(virginica[2]),
            np.average(virginica[3]),
        ],
        "plant": "virginica",
    }

    df_setosa = pd.DataFrame(avg_setosa)
    df_versicolor = pd.DataFrame(avg_versicolor)
    df_virginica = pd.DataFrame(avg_virginica)

    average_df = pd.concat((df_setosa, df_versicolor, df_virginica))

    fig1 = px.bar(average_df, x="attributes", y="average", color="plant")
    fig1.show()

    # Scatter plot for sepal width vs sepal length
    df = px.data.iris()
    fig2 = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        size="petal_length",
        hover_data=["petal_width"],
    )
    fig2.show()

    # Scatter plot for sepal length vs sepal width
    fig3 = px.scatter(
        df,
        x="petal_width",
        y="petal_length",
        color="species",
        size="sepal_length",
        hover_data=["sepal_width"],
    )
    fig3.show()

    data.columns = columns

    # Violin plot
    fig4 = px.violin(
        data,
        y="petal_len",
        x="petal_wid",
        color="class",
        box=True,
        points="all",
        hover_data=data.columns,
    )
    fig4.show()

    # Histogram
    fig5 = px.histogram(data, x="petal_len", color="class", nbins=10)
    fig5.show()

    # Random classifier model using iris dataset

    # Add response columns to iris dataframe
    data["is_setosa"] = (data["class"] == "Iris-setosa") * 1
    data["is_versicolor"] = (data["class"] == "Iris-versicolor") * 1
    data["is_virginica"] = (data["class"] == "Iris-virginica") * 1

    # ML Model for setosa class
    X_orig_setosa = data[columns[0:4]].values
    y_setosa = data["is_setosa"].values

    # StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_orig_setosa)
    x_orig_setosa = scaler.transform(X_orig_setosa)

    # Fit the features to a random forrest
    random_forest_setosa = RandomForestClassifier(random_state=1234)
    random_forest_setosa.fit(x_orig_setosa, y_setosa)
    test_data = data[columns[0:4]]
    x_orig_setosa_test = test_data.values
    trans_test_data = scaler.transform(x_orig_setosa_test)
    prediction_setosa = random_forest_setosa.predict(trans_test_data)
    probability_setosa = random_forest_setosa.predict_proba(trans_test_data)
    print(f"Probability: {probability_setosa}")
    print(f"Predictions: {prediction_setosa}")

    # Fit the features to k nearest neighbor
    neigh_setosa = KNeighborsClassifier(n_neighbors=3)
    neigh_setosa.fit(x_orig_setosa, y_setosa)
    neight_setosa_pred = neigh_setosa.predict(trans_test_data)
    print(f"Predictions_neigh_setosa: {neight_setosa_pred}")

    # Fit the features to bayes classifier
    clf_setosa = GaussianNB()
    clf_setosa.fit(x_orig_setosa, y_setosa)
    clf_setosa_pred = clf_setosa.predict(trans_test_data)
    print(f"Predictions_clf_setosa: {clf_setosa_pred}")

    # As pipeline using random forrest
    pipeline_set = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline_set.fit(x_orig_setosa, y_setosa)

    probability_set_pip = pipeline_set.predict_proba(trans_test_data)
    prediction_set_pip = pipeline_set.predict(trans_test_data)
    print(f"Probability_pip: {probability_set_pip}")
    print(f"Predictions_pip: {prediction_set_pip}")

    # Mean Response Plot Function
    mean_response_plot("sepal_len", "is_setosa")
    mean_response_plot("sepal_wid", "is_setosa")
    mean_response_plot("petal_len", "is_setosa")
    mean_response_plot("petal_wid", "is_setosa")

    mean_response_plot("sepal_len", "is_versicolor")
    mean_response_plot("sepal_wid", "is_versicolor")
    mean_response_plot("petal_len", "is_versicolor")
    mean_response_plot("petal_wid", "is_versicolor")

    mean_response_plot("sepal_len", "is_virginica")
    mean_response_plot("sepal_wid", "is_virginica")
    mean_response_plot("petal_len", "is_virginica")
    mean_response_plot("petal_wid", "is_virginica")


if __name__ == "__main__":
    sys.exit(main())
