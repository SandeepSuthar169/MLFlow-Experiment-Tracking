import mlflow
import mlflow.sklearn
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000") # mlflow server traking of app.py file 

wine = load_wine()
X= wine.data
y= wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

max_depth = 20
n_estimators = 10

mlflow.autolog()
mlflow.set_experiment('sample_Exp_2')  # new expremente traking 

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth,
                                n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)  # create confusion matris and logging 
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Metrics')

    plt.savefig('Confustion-matrix.png')  # save confusion matris picture 


    mlflow.log_artifact(__file__)    # this file logging to exprementent
    # tags
    mlflow.set_tags({'Author': "Sandeep", "Project": "wine classificaton"})


    print(accuracy)