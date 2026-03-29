print("Exam Marks Predictor")
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
Y=np.array([35,41,53,56,66,72,75,86,91,96])
model=LinearRegression()
model.fit(X,Y)
predicted_marks = model.predict(X)
plt.scatter(X,Y)
plt.plot(X,predicted_marks)
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Study Hours V/S Exam Marks")
plt.show()
a=float(input("Enter hours study in a day: "))
b=model.predict(np.array([[a]]))
print(f"Predicted marks for {a} hours of study:{b[0]:}")