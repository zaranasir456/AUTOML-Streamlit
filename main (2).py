
## Core
import pandas as pd
import os
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from mlxtend.evaluate import mcnemar
from sklearn.model_selection import RandomizedSearchCV
import plotly.graph_objects as go

import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

## Custome Components
import sweetviz as sv


def st_display_sweetviz(report_html, width=1000, height=500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)


## EDA
import pandas as pd
import numpy as np
import codecs
from pandas_profiling import ProfileReport

## Data Visualization
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
# COLOR = "black"
# BACKGROUND_COLOR = "#fff"
import seaborn as sns

## Machine Learning
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC




def grad(X, y, w, lr, n_iter):
    losses = []
    for i in range(n_iter):
        error = np.dot(X, w) - y
        w -= (lr/y.shape[0])*np.dot(error.T, X).T

        loss = (1/2*y.shape[0])*np.dot((y-np.dot(X, w)).T, (y-np.dot(X, w)))[0][0]
        losses.append(loss)
    return losses, w


def main():
    """AutoML Web App Tool with Streamlit"""

    st.title("AutoML")
    st.sidebar.image("ml1.png")

    #     activities = ["EDA", "Plot", "Model Building", "About"]
    #     choice = st.sidebar.selectbox("Select Activity", activities)

    #     dark_theme = st.sidebar.checkbox("Dark Theme", False)
    menu = ["Home", "Pandas Profile", "Plot", "Model Building", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.subheader('Learning Parameters')
    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1, 3), 1)
    st.sidebar.number_input('Step size for max_features', 1)

    if choice == 'Home':
        #
        st.markdown(
            '**Data Analysis, Visualization** and Machine Learning **Model Building** in an interactive **WebApp** for Data Scientist/Data Engineer/Business Analyst. \n\nAutoML WebApp built with **Streamlit framework** using **Pandas** and **Numpy** for Data Analysis, **Matplotlib** and **Seaborn** for Data Visualization, **SciKit-Learn** for Machine Learning Model.')
        #         st.markdown('**Demo URL**: https://automlwebapp.herokuapp.com/')

    if choice == 'Pandas Profile':
        st.subheader("Automated EDA with Pandas Profile")

        data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            df = (df - df.min())/(df.max()-df.min())
            st.dataframe(df.head())
            profile = ProfileReport(df)
            st_profile_report(profile)


    elif choice == 'Plot':
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            df = (df - df.min())/(df.max()-df.min())
            st.dataframe(df.head())

        if st.checkbox("Correlation with Seaborn"):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

        if st.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            columns_to_plot = st.selectbox("Select 1 Column", all_columns)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()

        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

            ## Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            ## Custom Plot
            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()



    elif choice == 'Model Building':
        st.subheader("Building Ml Model")

        data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            df = df.dropna()
            #df = df.pop(["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2"])
            #print(df.head)
            #df = (df - df.min())/(df.max()-df.min())
            st.dataframe(df.head())

            ## Model Building
            X = df.iloc[:, 0:-1].values

            X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
            Y = df.iloc[:, -1].values

            seed = 7

            ## Model
            models = []
            models.append(("LR", LinearRegression()))
            #models.append(("KNN", KNeighborsClassifier(n_neighbors=3)))
            ## Evaluate each model in turn

            ## List
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.3, random_state=0)
            trainy = np.expand_dims(trainy, 1)
            testy = np.expand_dims(testy, 1)
            print(trainX.shape, trainy.shape)
            #knn = models[1][1].fit(trainX, trainy)
            X_ = np.concatenate((np.ones((trainX.shape[0], 1)), trainX), axis=1)
            w = np.zeros((X_.shape[1], 1))
            losses, w = grad(X_, trainy, w, 0.001, 4000)

            y_pred = np.dot(np.concatenate((np.zeros((testy.shape[0], 1)), testX), axis=1), w)
            mean_pred = np.sum(y_pred)/y_pred.shape[0]*np.ones(y_pred.shape)
            acc = 1-(np.dot((testy-y_pred).T, (testy-y_pred))[0][0]/np.dot((testy-mean_pred).T, (testy-mean_pred))[0][0])

            std = (np.dot((testy-np.dot(np.concatenate((np.ones((testy.shape[0], 1)), testX), axis=1), w)).T, (testy-np.dot(np.concatenate((np.ones((testy.shape[0], 1)), testX), axis=1), w)))[0][0]/(testy.shape[0]-2))**0.5

            name = "LR"
            #kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
            #cv_results = model_selection.cross_val_score(model, testX, testy)
            model_names.append(name)
            model_mean.append(acc)
            model_std.append(std)

            accuracy_results = {"model_name": name, "model_accuracy": acc,
                                "standard_deviation": std}
            all_models.append(accuracy_results)

            # if st.checkbox("Metrics as KNN"):
                # accuracy = sklearn.metrics.accuracy_score(knn, testX, testy, scoring='wrong_choice')
                # f1 = sklearn.metrics.f1_score(knn, testX, testy, average='weighted')
                # precision = sklearn.metrics.precision_score(knn, testX, testy, average='weighted')

            if st.checkbox("Metrics as Linear Regression"):
                #trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.3, random_state=0)

                # conf_matrix = confusion_matrix(y_true=testy, y_pred=y_pred)
                # fig, ax = plt.subplots()
                # for i in range(conf_matrix.shape[0]):
                #     for j in range(conf_matrix.shape[1]):
                #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='centre', ha='centre', size='x-large')
                plt.plot(losses, c='r')
                plt.xlabel('Iterations', fontsize=11)
                plt.ylabel('Error Values', fontsize=11)
                plt.title('Error vs. Iterations', fontsize=11)
                plt.show()
                st.pyplot()
                # mae = sklearn.metrics.mean_absolute_error(Y=testy, y_pred=y_pred)
                # mse = sklearn.metrics.mean_squared_error(Y=testy, y_pred=y_pred) #default true
                # rmse = sklearn.metrics.mean_squared_error(Y=testy, y_pred=y_pred, squared=False)

            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std),
                columns=["Model Name", "Model Accuracy", "Standard Deviation"]))

    elif choice == "About":
        st.header("About Author")
        #         st.markdown('**Data Analysis, Visualization** and Machine Learning **Model Building** in an interactive **WebApp** for Data Scientist/Data Engineer/Business Analyst.  \n\nThe purpose of this app is to create a **quick Business Insights**.  \n\nAutoML WebApp built with **Streamlit framework** using **Pandas** and **Numpy** for Data Analysis, **Matplotlib** and **Seaborn** for Data Visualization, **SciKit-Learn** for Machine Learning Model.')
        # #         st.markdown('**Demo URL**: https://automlwebapp.herokuapp.com/')
        #         st.header("Silent Features")
        #         st.markdown('* User can browse or upload file(Dataset) in .csv or .txt format.  \n* User can get the details of dataset like No. of rows & Columns, Can View Column list, Select Columns with rows to show, Dataset Summary like count, mean, std, min and max values.  \n* Several Data Visualizations like Correlation with HeatMap, PieChart and Plots like Area, Bar, Line, Box, KDE.  \n* User can built Models like LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, SVC.  \n* Model Evaluation with Accuracy, Mean and Standard Deviation.')
        #         st.header("Author")
        st.markdown(
            "Hi, there! You are bad at PROGRAMMING? Don't worry at all. Here is the AUTOML Webapp generated by passionate AI Students. In this webapp, you don't need to code anything. Moreover, there are customization options available as well.")
        st.markdown('**Github**: https://github.com/zaranasir456   ')
        #         st.markdown('**GitHub**: https://github.com/ravivarmathotakura')
        st.markdown('**LinkedIn**: https://www.linkedin.com/in//zara-nasir-28a99a20a')

#     else:
#         st.subheader("Home")
#         html_temp = """
#         <div style = "background-color:royalblue;padding:10px;border-radius:10px">
#         <h1 style = "color:white;text-align:center;">Simpe EDA App with Streamlit Components
#         </div>
#         """

#         #components.html("<p style='color:red'> Streamlit App </p>")
#         components.html(html_temp)

if __name__ == '__main__':
    main()
