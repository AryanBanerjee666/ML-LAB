import pandas as pd

data={
    "Name":["ARYAN","BIKRAM","GOURAB","RAHUL","SOURAV","NEHA","RIYA","AMIT","ASHOK","NAYAN"],
    "Roll no.": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [98,95,77,85,65,45,85,98,65,75],
    "Attendance":[98,78,95,62,75,84,65,75,89,95]
}
df=pd.DataFrame(data)

def assign_grade(Marks):
    if Marks>=90:
        return'A'
    elif Marks>=80:
        return 'N'
    elif Marks>=70:
        return 'C'
    elif Marks>=60:
        return 'D'
    else :
        return 'E'



df['Grade'] = df['Marks'].apply(assign_grade)
print(df)
