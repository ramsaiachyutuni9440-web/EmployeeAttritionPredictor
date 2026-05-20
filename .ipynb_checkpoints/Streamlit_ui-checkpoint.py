import streamlit as st
import pandas as pd
import joblib

model = joblib.load("attrition_model.pkl")

st.title("IBM HR Employee Attrition Predictor")

st.header("Enter Employee Details")

Age = st.slider("Age", 18, 60, 30)
DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
Education = st.selectbox("Education", [1, 2, 3, 4, 5])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
HourlyRate = st.number_input("Hourly Rate", 30, 100, 65)
JobInvolvement = st.selectbox("Job Involvement (1=Low, 4=High)", [1, 2, 3, 4])
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
JobSatisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
MonthlyRate = st.number_input("Monthly Rate", 2000, 27000, 14000)
NumCompaniesWorked = st.slider("Num Companies Worked", 0, 9, 2)
OverTime = st.selectbox("OverTime", ["Yes", "No"])
PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 14)
PerformanceRating = st.selectbox("Performance Rating", [3, 4])
RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
TotalWorkingYears = st.slider("Total Working Years", 0, 40, 8)
TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 2)
WorkLifeBalance = st.selectbox("Work Life Balance (1=Low, 4=High)", [1, 2, 3, 4])
YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
YearsInCurrentRole = st.slider("Years In Current Role", 0, 18, 3)
YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 3)
BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
JobRole = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
    }])
    
    input_df["OverTime"] = input_df["OverTime"].map({"Yes": 1, "No": 0})
    input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})
    input_df["BusinessTravel"] = input_df["BusinessTravel"].map({
            "Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2
    })
    input_df["Department"] = input_df["Department"].map({
        "Human Resources": 0, "Research & Development": 1, "Sales": 2
    })
    input_df["EducationField"] = input_df["EducationField"].map({
        "Human Resources": 0, "Life Sciences": 1, "Marketing": 2,
        "Medical": 3, "Other": 4, "Technical Degree": 5
    })
    input_df["JobRole"] = input_df["JobRole"].map({
            "Healthcare Representative": 0, "Human Resources": 1,
            "Laboratory Technician": 2, "Manager": 3,
            "Manufacturing Director": 4, "Research Director": 5,
            "Research Scientist": 6, "Sales Executive": 7, "Sales Representative": 8
    })
    input_df["MaritalStatus"] = input_df["MaritalStatus"].map({
            "Divorced": 0, "Married": 1, "Single": 2
    })
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    
    if prediction == 1:
        st.error(f"⚠️ Employee is likely to leave — Attrition Risk: {probability:.1f}%")
    else:
        st.success(f"✅ Employee is likely to stay — Attrition Risk: {probability:.1f}%")