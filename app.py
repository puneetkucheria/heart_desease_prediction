from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

#import model
model = joblib.load('model_hdpred.pkl')

class Heart_Desease_parms(BaseModel):
    BMI: float
    Smoking: int
    AlcoholDrinking: int
    Stroke: int
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: int
    Sex: int
    PhysicalActivity: int
    SleepTime: float
    Asthma: int
    KidneyDisease: int
    SkinCancer: int
    AgeCategory_1824: bool
    AgeCategory_2529: bool
    AgeCategory_3034: bool
    AgeCategory_3539: bool
    AgeCategory_4044: bool
    AgeCategory_4549: bool
    AgeCategory_5054: bool
    AgeCategory_5559: bool
    AgeCategory_6064: bool
    AgeCategory_6569: bool
    AgeCategory_7074: bool
    AgeCategory_7579: bool
    AgeCategory_80_or_older: bool
    Race_American_Indian_Alaskan_Native: bool
    Race_Asian: bool
    Race_Black: bool
    Race_Hispanic: bool
    Race_Other: bool
    Race_White: bool
    Diabetic_: bool
    Diabetic_No: bool
    Diabetic_No_borderline_diabetes: bool
    Diabetic_Yes: bool
    Diabetic_Yes_during_pregnancy: bool
    GenHealth_Excellent: bool
    GenHealth_Fair: bool
    GenHealth_Good: bool
    GenHealth_Poor: bool
    GenHealth_Very_good: bool

    
@app.get('/')
async def root():
    return{"status":"Online"}

@app.post('/HeartDisease/')
async def heart_desease(parms:Heart_Desease_parms):
    # print("Parameters : " + str(parms.step))
    print(parms)
    re = model.predict([[parms.BMI,parms.Smoking,parms.AlcoholDrinking, parms.Stroke, parms.PhysicalHealth, parms.MentalHealth\
                         ,parms.DiffWalking, parms.Sex, parms.PhysicalActivity, parms.SleepTime, parms.Asthma, parms.KidneyDisease\
                            ,parms.SkinCancer, parms.AgeCategory_1824, parms.AgeCategory_2529, parms.AgeCategory_3034, parms.AgeCategory_3539\
                                ,parms.AgeCategory_4044, parms.AgeCategory_4549, parms.AgeCategory_5054, parms.AgeCategory_5559, parms.AgeCategory_6064\
                                    ,parms.AgeCategory_6569, parms.AgeCategory_7074, parms.AgeCategory_7579, parms.AgeCategory_80_or_older\
                                        ,parms.Race_American_Indian_Alaskan_Native, parms.Race_Asian, parms.Race_Black, parms.Race_Hispanic\
                                            ,parms.Race_Other, parms.Race_White, parms.Diabetic_, parms.Diabetic_No, parms.Diabetic_No_borderline_diabetes\
                                                ,parms.Diabetic_Yes, parms.Diabetic_Yes_during_pregnancy, parms.GenHealth_Excellent, parms.GenHealth_Fair\
                                                    ,parms.GenHealth_Good, parms.GenHealth_Poor, parms.GenHealth_Very_good]])
    # print(re)
    return {"Heart Desease pridiction ": str(re[0])}