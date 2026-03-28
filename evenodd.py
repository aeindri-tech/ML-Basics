from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("Even V/S Odd Numbers Predictor")
X= [[i%2] for i in range (1,501)]
Y=[0 if i%2==0 else 1 for i in range(1,501)]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train,Y_train)
a=int(input("Enter a number: "))
b=(clf.predict([[a%2]]))
if b[0]==0:
    print("The number is even")
else:
    print("The number is odd")
predictions=clf.predict(X_test)
score=accuracy_score(Y_test,predictions)
print(f"The prediction is accurate by {score*100}%")


