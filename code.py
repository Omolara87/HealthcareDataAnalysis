import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime

# Load data
patients = pd.read_csv("patient_data.csv", parse_dates=["DOB", "Admission_Date", "Discharge_Date"])
hospitals = pd.read_csv("hospital_data.csv")
treatments = pd.read_csv("treatment_data.csv")

# 1. Merge data
df = patients.merge(hospitals, on="Hospital_ID", how="left").merge(treatments, on="Condition", how="left")

# 2. Handle missing/inconsistent data
df.fillna(method='ffill', inplace=True)
df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])

# 3. Calculate age
today = pd.to_datetime("2024-01-01")
df["Age"] = (today - df["DOB"]).dt.days // 365

# 4. Categorize by age groups
bins = [0, 18, 35, 50, 65, 100]
labels = ["Child", "Young Adult", "Adult", "Middle Aged", "Senior"]
df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

# 5. Categorize medical conditions already done in data

# 6. Admission/discharge trends
df["Admission_Month"] = df["Admission_Date"].dt.month
df["Discharge_Month"] = df["Discharge_Date"].dt.month

admission_trend = df.groupby("Admission_Month")["Patient_ID"].count()
discharge_trend = df.groupby("Discharge_Month")["Patient_ID"].count()

# 7. Length of stay
df["Length_of_Stay"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days

avg_stay_condition = df.groupby("Condition")["Length_of_Stay"].mean()
avg_stay_department = df.groupby("Department")["Length_of_Stay"].mean()

# 8. Readmission rates
readmission_rate = df["Readmitted"].mean()
readmission_by_condition = df.groupby("Condition")["Readmitted"].mean()

# 9. Segment by treatment outcome
outcome_segment = df.groupby("Outcome")["Patient_ID"].count()

# 10. Treatment effectiveness analysis
effectiveness_summary = df.groupby("Treatment_Type")["Effectiveness_Rate"].mean()

# 11. Pie chart for gender
df["Gender"].value_counts().plot(kind='pie', autopct='%1.1f%%', title="Patient Gender Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("patient_gender_pie.png")
plt.close()

# 12. Pie chart for age groups
df["Age_Group"].value_counts().plot(kind='pie', autopct='%1.1f%%', title="Age Group Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("age_group_pie.png")
plt.close()

# 13. Line plots for admission/discharge trends
plt.plot(admission_trend.index, admission_trend.values, marker='o', label='Admissions')
plt.plot(discharge_trend.index, discharge_trend.values, marker='x', label='Discharges')
plt.title("Monthly Admissions and Discharges")
plt.xlabel("Month")
plt.ylabel("Number of Patients")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("admission_discharge_trend.png")
plt.close()

# 14. Box plot for length of stay
sns.boxplot(x="Condition", y="Length_of_Stay", data=df)
plt.xticks(rotation=45)
plt.title("Length of Stay by Condition")
plt.tight_layout()
plt.savefig("length_of_stay_boxplot.png")
plt.close()

# 15. Dashboard placeholder — print key metrics (can be adapted to Streamlit)
print("\n=== Key Metrics ===")
print("Readmission Rate:", round(readmission_rate * 100, 2), "%")
print("Average Stay by Condition:\n", avg_stay_condition)
print("Average Stay by Department:\n", avg_stay_department)
print("Treatment Effectiveness:\n", effectiveness_summary)

# 16. High-risk patient identification (based on multiple stays, poor outcome)
high_risk_patients = df[df["Outcome"] == "Deteriorated"]

# 17. Recommend interventions (manual logic)
interventions = df[df["Readmitted"] == 1].groupby("Condition")["Treatment_Type"].agg(lambda x: x.value_counts().index[0])

# 18. Export results
df.to_csv("full_patient_analysis.csv", index=False)
avg_stay_condition.to_csv("avg_stay_by_condition.csv")
readmission_by_condition.to_csv("readmission_by_condition.csv")

# 19. Connect to SQL database (demo)
conn = sqlite3.connect(":memory:")
df.to_sql("patients", conn, index=False, if_exists="replace")

# 20. Documentation (can also be saved to a Markdown or text file)
print("\nTop Conditions with Readmissions:\n", readmission_by_condition.sort_values(ascending=False).head())
print("\nSuggested Interventions:\n", interventions)

