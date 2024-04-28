from data_processing_features import get_heart_data, convert_num_col, map_cat_col, replace_missing_data, get_dummies
from model_building import train_test_split_and_features, smote_data, fit_and_evaluate_model
from xgboost import XGBClassifier
import joblib

df = get_heart_data()
print("data loaded")

num_col = ['BMI','PhysicalHealth', 'MentalHealth','SleepTime']
cat_col = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
pre_col = ['HeartDisease']

df = convert_num_col(df, num_col=num_col)
print("num col converted")
df = map_cat_col(df)
print("cat col mapped")
df = replace_missing_data(df)
print("missing data replaced")
df = get_dummies(df)
print("dummies created")

# train test split and resampled
x_train, x_test, y_train, y_test,features = train_test_split_and_features(df)
print("data splitted")
x_res, y_res = smote_data(x_train=x_train, y_train=y_train)
print("data resampled")

# fit and evaluate model
xgb =  XGBClassifier(random_state = 0, n_estimators = 150, learning_rate = 0.3, max_depth = 10, subsample = 0.8, colsample_bytree = 1)
model = fit_and_evaluate_model(x_res, x_test, y_res, y_test,xgb)
print("completed")
# model.save_model('model_heart_desease.h5')

joblib.dump(model , 'model_hdpred.pkl')
print('model saved')