
# Import Other modeling libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from imblearn.over_sampling import SMOTE
# from sklearn.datasets import make_classification
# from collections import Counter

def train_test_split_and_features(data):
    y = data['HeartDisease']
    x = data.drop('HeartDisease',axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle = False, random_state = 0)
    # print(x.head(5))
    print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test, features

def smote_data(x_train, y_train):
    #Handling Imbalance Data
    # Define the percentage of oversampling 
    sampling_strategy = 0.9  # number between 0 to 1 
    # after resampling minority class would be 20% of majority class

    # Apply SMOTE with specified oversampling percentage
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res

def fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb):
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb