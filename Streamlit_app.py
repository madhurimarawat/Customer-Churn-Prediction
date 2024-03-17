# In this program we will apply various ML algorithms to the built in datasets in scikit-learn

# Importing required Libraries
# Importing Numpy
import numpy as np
# To read csv file
import pandas as pd
# For splitting between training and testing
from sklearn.model_selection import train_test_split
# Importing Algorithm for Simple Vector Machine
from sklearn.svm import SVC
# Importing Knn algorithm
from sklearn.neighbors import KNeighborsClassifier
# Importing  Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# Importing Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# Importing Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
# Importing Linear and Logistic Regression
from sklearn.linear_model import LogisticRegression
# Importing accuracy score
from sklearn.metrics import accuracy_score
# Importing PCA for dimension reduction
from sklearn.decomposition import PCA
# For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For model deployment
import streamlit as st

# Reading Data
data = pd.read_csv("Preprocessed_Customer_Prediction.csv")


# Creating Multipage Streamlit Application for showing each stage

#<--------------------------------------------------- Step 1 Exploratory Data Analysis --------------------------------------------------->

def Step_1_EDA(df):

    st.title("Exploratory Data Analysis")

    st.write("Columns of Dataset are:\n", df.columns)

    st.write("\nShape of Dataset is:", df.shape)

    # Total Count of Datatypes
    print("Datatype of Dataset is:\n")
    df.info()

    # According to Columns
    data_types = df.dtypes

    print("\nDatatypes of Columns is:\n")
    print(data_types)

    Numerical_Columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    st.subheader("Numerical Columns Plot")

    # Plotting histograms and boxplots for numerical columns
    for i in Numerical_Columns:

        if df[i].dtype == 'int64' or df[i].dtype == 'float64':

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Plotting histogram
            sns.histplot(df[i], kde=True, ax=axs[0])

            # Adjusting font weight for axis labels, title, and ticks
            axs[0].set_xlabel(i, fontweight='bold')
            axs[0].set_ylabel('Count', fontweight='bold')
            axs[0].set_title(f"Distribution of {i}", fontweight='bold')
            plt.setp(axs[0].get_xticklabels(), fontweight='bold')
            plt.setp(axs[0].get_yticklabels(), fontweight='bold')

            sns.boxplot(df[i], ax=axs[1])

            # Adjusting font weight for axis labels, title, and ticks
            axs[1].set_xlabel(i, fontweight='bold')
            axs[1].set_ylabel('Count', fontweight='bold')
            axs[1].set_title(f"Distribution of {i}", fontweight='bold')
            plt.setp(axs[1].get_xticklabels(), fontweight='bold')
            plt.setp(axs[1].get_yticklabels(), fontweight='bold')

            # Printing the plot in streamlit website
            st.pyplot(fig)

    # Plotting count plots for categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                           'PaymentMethod']

    st.subheader("Categorical Columns Plot")

    fig, axs = plt.subplots(8, 2, figsize=(20, 40))

    plt.subplots_adjust(hspace=0.5)

    for i, col in enumerate(categorical_columns):
        sns.countplot(x=df[col], data=df, ax=axs[i // 2, i % 2], hue='Churn')

        # Adjusting font weight for axis labels, title, and ticks
        axs[i // 2, i % 2].set_xlabel(col, fontweight='bold')
        axs[i // 2, i % 2].set_ylabel('Count', fontweight='bold')
        axs[i // 2, i % 2].set_title(f"Distribution of {col}", fontweight='bold')
        plt.setp(axs[i // 2, i % 2].get_xticklabels(), fontweight='bold')
        plt.setp(axs[i // 2, i % 2].get_yticklabels(), fontweight='bold')

        # Fixing the legend box
        axs[i // 2, i % 2].legend(loc='upper right', frameon=True, fontsize='large')

    plt.tight_layout()

    # Printing the plot in streamlit website
    st.pyplot(fig)

#<------------------------------------------ Step 2 Feature Engineering------------------------------------------------------------------->

def Step_2_Feature_Engineering():

    st.title("Feature Engineering")

    st.info("Feature engineering is the process of transforming raw data into meaningful features that enhance the "
            "performance of machine learning models, "
            "often involving tasks such as dimensionality reduction, creating new features, and handling missing values.")

    st.subheader("Here in this dataset, customer Id column was removed as it was not useful")

    st.subheader("After this the target Variable Churn class was highly imbalanced so it was balanced")

    st.write("Value Counts of Target Column after Feature Engineering is:", data['Churn'].value_counts())

#<------------------------------------------ Step 3 Data Preprocessing --------------------------------------------------------------------->

def Step_3_Data_Preprocessing(df):

    st.title("Data Preprocessing")

    st.info("Data preprocessing involves transforming raw data into a clean, structured format suitable for analysis.")

    st.subheader("Here in this dataset, Encoding of Categorical Columns was done")

    st.subheader("After this the normalization and Scaling was done for column monthly and total charges")

    st.write("Descriptive Statistics of Dataset after Preprocessing is:", df.describe())

    import warnings
    warnings.filterwarnings("ignore")

    corr_mat = df.corr()

    st.write("\n")

    st.subheader("Correlation Plot of Dataset\n\n")

    # Setting Figure Size
    fig = plt.figure(figsize=(19, 10))

    sns.heatmap(corr_mat, annot=True, cmap='viridis')

    plt.title("Heat Map/ Correlation Plot for Customer Churn Prediction Dataset", fontweight = 'bold', fontsize = 22, pad=20)
    plt.xticks(fontweight = 'bold')
    plt.yticks(fontweight='bold')

    # Printing the plot in streamlit website
    st.pyplot(fig)

#<------------------------------------------ Step 4 Model Development -------------------------------------------------------------------->

def Step_4_Model_Development():

    st.title("Model Development")

    st.info("Model development involves training and fine-tuning machine "
            "learning algorithms to make predictions or classifications based on input data.")

    st.write("The Following Model were trained for this dataset:\n")

    # List of models
    models = ["KNN", "SVM", "Decision Tree", "Naive Bayes", "Random Forest", "Logistic Regression"]

    # One-liner descriptions for each model
    descriptions = {
        "KNN": "K-Nearest Neighbors is a simple, instance-based learning algorithm.",
        "SVM": "Support Vector Machine is a powerful supervised learning algorithm.",
        "Decision Tree": "Decision Tree is a versatile algorithm known for its interpretability.",
        "Naive Bayes": "Naive Bayes is a probabilistic classifier based on Bayes' theorem.",
        "Random Forest": "Random Forest is an ensemble learning method that builds multiple decision trees.",
        "Logistic Regression": "Logistic Regression is a linear model used for binary classification."
    }

    # Creating a dataframe from the list
    df_models = pd.DataFrame(models, columns=["Models"], index=range(1, 7))

    # Adding one-liner descriptions to the dataframe
    df_models["Description"] = df_models["Models"].map(descriptions)

    # Display the DataFrame as a Markdown table
    # To successfully run this we need to install tabulate
    st.markdown(df_models.to_markdown(index=False), unsafe_allow_html=True)

#<------------------------------------------ Step 5 Model Evaluation -------------------------------------------------------------------->

# Adding Parameters so that we can select from various parameters for classifier
def add_parameter_classifier(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Deciding parameters based on algorithm
    # Adding paramters for SVM
    if algorithm == 'SVM':

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        # Kernel is the arguments in the ML model
        # Polynomial ,Linear, Sigmoid and Radial Basis Function are types of kernals which we can add
        kernel_custom = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
        # Adding in dictionary
        params['C'] = c_regular
        params['kernel'] = kernel_custom
        if kernel_custom == 'linear':
            st.sidebar.info("SVM is Slow for this kernel as the dataset is very large.Try with other kernels speed will be improved.")

    # Adding Parameters for KNN
    elif algorithm == 'KNN':

        # Adding number of Neighbour in Classifier
        k_n = st.sidebar.slider('Number of Neighbors (K)', 1, 20)
        # Adding in dictionary
        params['K'] = k_n
        # Adding weights
        weights_custom = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
        # Adding to dictionary
        params['weights'] = weights_custom

    elif algorithm == 'Naive Bayes':
        st.sidebar.info("This is a simple Algorithm. It doesn't have Parameters for Hyper-tuning.")

    # Adding Parameters for Decision Tree
    elif algorithm == 'Decision Tree':

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding criterion
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
        # Adding splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        # Taking random state
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['criterion'] = criterion
        params['splitter'] = splitter

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Random Forest
    elif algorithm == 'Random Forest':

        # Taking max_depth
        max_depth = st.sidebar.slider('Max Depth', 2, 17)
        # Adding number of estimators
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 9)
        # Adding criterion
        # mse is for regression- It is used in RandomForestRegressor
	    # mse will give error in classifier so it is removed
        criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))
        # Adding to dictionary
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion

        # Exception Handling using try except block
        # Because we are sending this input in algorithm model it will show error before any input is entered
        # For this we will do a default random state till the user enters any state and after that it will be updated
        try:
            random = st.sidebar.text_input("Enter Random State")
            params['random_state'] = int(random)
        except:
            params['random_state'] = 4567

    # Adding Parameters for Logistic Regression
    else:

        # Adding regularization parameter from range 0.01 to 10.0
        c_regular = st.sidebar.slider('C (Regularization)', 0.01, 10.0)
        params['C'] = c_regular
        # Taking fit_intercept
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ('True', 'False'))
        params['fit_intercept'] = bool(fit_intercept)
        # Taking Penalty only l2 and None is supported
        penalty = st.sidebar.selectbox("Penalty", ('l2', None))
        params['penalty'] = penalty
        # Taking n_jobs
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params['n_jobs'] = n_jobs

    return params


# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_classifier(algorithm, params):

    if algorithm == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'], weights=params['weights'])

    elif algorithm == 'SVM':
        return SVC(C=params['C'], kernel=params['kernel'])

    elif algorithm == 'Decision Tree':
        return DecisionTreeClassifier(
            criterion=params['criterion'], splitter=params['splitter'],
            random_state=params['random_state'])

    elif algorithm == 'Naive Bayes':
        return GaussianNB()

    elif algorithm == 'Random Forest':
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      criterion=params['criterion'],
                                      random_state=params['random_state'])

    else:
        return LogisticRegression(fit_intercept=params['fit_intercept'],
                                  penalty=params['penalty'], C=params['C'], n_jobs=params['n_jobs'])


# Now we will write the dataset information
def info(data_name, algorithm, X, Y):

    st.write(f"## Classification {data_name} Dataset")
    st.write(f'Algorithm is : {algorithm + " " + "Classifier"}')

    # Printing shape of data
    st.write('Shape of Dataset is: ', X.shape)
    st.write('Number of classes: ', len(np.unique(Y)))
    # Making a dataframe to store target name and value
    df = pd.DataFrame({"Target Value": list(np.unique(Y)),
                       "Target Name": ['Not Fraud', 'Fraud']})
    # Display the DataFrame without index labels
    st.write('Values and Name of Classes')

    # Display the DataFrame as a Markdown table
    # To successfully run this we need to install tabulate
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
    st.write("\n")


def Step_5_Model_Evaluation(data):

    # Giving Title
    st.title("Model Evaluation")

    # Now we are making a select box for dataset
    data_name = "Customer Churn Prediction"

    # The Next is selecting algorithm
    # We will display this in the sidebar
    algorithm = st.sidebar.selectbox("Select Supervised Learning Algorithm",
                                     ("KNN", "SVM", "Decision Tree", "Naive Bayes", "Random Forest",
                                      "Logistic Regression"))

    # Now after this we need to split between input and output
    # Defining Input and Output
    # Separating as input and output
    X, Y = data.drop(['Churn'], axis=1), data['Churn']

    # Now splitting into Testing and Training data
    # It will split into 80 % training data and 20 % Testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    # Ensuring data contiguity using np.ascontiguousarray() to resolve the 'c_contiguous' attribute issue
    # This is done because certain scikit learn function expect contiguous data
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)
    y_test = np.ascontiguousarray(y_test)

    # Calling Function based on algorithm
    params = add_parameter_classifier(algorithm)


    # Choosing algorithm
    # Calling Function based on classifier
    algo_model = model_classifier(algorithm,params)

    # Calling function to print Dataset Information
    info(data_name, algorithm, X, Y)

    # Training algorithm
    algo_model.fit(x_train,y_train)

    # Now we will find the predicted values
    predict = algo_model.predict(x_test)

    # Finding Accuracy
    # Evaluating/Testing the model
    # For all algorithm we will find accuracy
    st.write("Training Accuracy is:",algo_model.score(x_train,y_train)*100)
    st.write("Testing Accuracy is:",accuracy_score(y_test,predict)*100)

    # Plotting Dataset
    # Since there are many dimensions, first we will do Principle Component analysis to do dimension reduction and then plot
    # Doing PCA for dimension reduction
    pca=PCA(3)
    x=pca.fit(x_test).transform(x_test)
    print("Transformed Data is:\n",x)
    print("\nShape of Transformed data is:",x.shape)

    # Plotting dataset
    fig = plt.figure()
    colors = ['lightblue','orange']
    # Adjusting alpha for transparency as there are a lot of overlapping points in the dataset
    sns.scatterplot(x=x[:,0], y=x[:,1], hue = predict,palette=sns.color_palette(colors),alpha=0.4)

    # Adding x and y labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Giving Title
    plt.title("Scatter Classification Plot of Dataset With Target Classes")

    # Printing the plot in streamlit website
    st.pyplot(fig)

#<------------------------------------------ Step 6 Predictions -------------------------------------------------------------------->

def Step_6_Predictions(data):

    st.title("Predictions")

    st.info("After Finding the best model we can give it to model and it will generate the target variable")

    st.write("This code snippet selects a specific tuple (row) from a DataFrame for prediction, drops the target "
             "variable column from the selected tuple, reshapes it to match the model's input requirements, and "
             "then predicts using the reshaped tuple. "
             "Finally, it prints the prediction and the actual target variable value for the sample tuple.")

    # Now after this we need to split between input and output
    # Defining Input and Output
    # Separating as input and output
    X, Y = data.drop(['Churn'], axis=1), data['Churn']

    # Now splitting into Testing and Training data
    # It will split into 80 % training data and 20 % Testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    # Ensuring data contiguity using np.ascontiguousarray() to resolve the 'c_contiguous' attribute issue
    # This is done because certain scikit learn function expect contiguous data
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)
    y_test = np.ascontiguousarray(y_test)

    from sklearn.ensemble import RandomForestClassifier

    # Define the best parameters
    best_params = {
        'criterion': 'gini',
        'max_depth': 10,
        'min_samples_leaf': 4,
        'min_samples_split': 10,
        'n_estimators': 200
    }

    # Initialize the RandomForestClassifier with best parameters
    rf_classifier = RandomForestClassifier(**best_params)

    # Train the classifier using input data X and target variable Y
    rf_classifier.fit(X, Y)

    # Input box to take index input from the user
    index = st.text_input("Enter the index of the row you want to predict:", "")

    # Check if the index is provided by the user
    if index:

        # Convert index to integer
        index = int(index)

        # Select the row from the DataFrame using the index
        sample_tuple = data.iloc[index]

        # Drop the target variable column from the selected tuple
        X_sample = sample_tuple.drop('Churn')

        # Reshape the sample tuple to match the model's input requirements
        X_sample_reshaped = X_sample.values.reshape(1, -1)

        # Make prediction using the sample tuple
        prediction = rf_classifier.predict(X_sample_reshaped)

        # Print the Actual Value
        st.write("Actual Value:", data.iloc[index]['Churn'])

        # Print the prediction
        st.write("Prediction:", prediction)


#<------------------------------------------ Step 7 Recommendations -------------------------------------------------------------------->

def Step_7_Recommendations():

    st.title("Recommendations for Reducing Customer Churn")

    st.markdown("### 1. Senior Citizen Benefits")
    st.markdown(
        "- As observed in the dataset, there is a lower usage among senior citizens, and the retention rate is also low. To increase customer retention, special benefits such as discounts, messages, or tech support can be provided to senior citizens.")

    st.markdown("### 2. Single User Benefits")
    st.markdown(
        "- More single users (without partners) are observed in the dataset. Providing special benefits like discounts or tech support tailored for single users can help in retaining them.")

    st.markdown("### 3. Events")
    st.markdown(
        "- Organizing informational events can help users utilize services more efficiently. Direct conversations with users may be more effective than emails.")

    st.markdown("### 4. Feedback")
    st.markdown(
        "- Collecting customer feedback can provide insights into the root causes of dissatisfaction. Conducting dissatisfaction analysis is crucial.")

    st.markdown("### 5. More Focus on Long-term Users")
    st.markdown(
        "- Users with a longer association with the company are more valuable. Offering extra benefits to users who have been using the service for an extended period, like 30-40 months, can reduce churn.")

    st.markdown("### 6. Utilizing ML Modeling")
    st.markdown(
        "- ML models can predict which users are more likely to churn. Providing additional benefits to such users based on model predictions can help reduce churn.")


# Main Function
if __name__ == "__main__":

    # Define navigation components
    nav = st.sidebar.radio("Step Navigation", ["Exploratory Data Analysis",
                                               "Feature Engineering", "Data Preprocessing",
                                               "Model Development", "Model Evaluation",
                                               "Predictions", "Recommendations"])

    # Render page content based on navigation
    if nav == "Exploratory Data Analysis":
        Step_1_EDA(data)

    elif nav == "Feature Engineering":
        Step_2_Feature_Engineering()

    elif nav == "Data Preprocessing":
        Step_3_Data_Preprocessing(data)

    elif nav == "Model Development":
        Step_4_Model_Development()

    elif nav == "Model Evaluation":
        Step_5_Model_Evaluation(data)

    elif nav == "Predictions":
        Step_6_Predictions(data)

    elif nav == "Recommendations":
        Step_7_Recommendations()



