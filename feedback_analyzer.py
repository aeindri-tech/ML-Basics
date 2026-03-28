import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
print("Student Feedback Analyzer")
Feedbacks=["This class was very boring",
           "This class sucks",
           "The lecture was confusing",
           "I did not enjoy today's class",
           "I did not understand the explanation",
           "I hate this class",
           "This topic is useless",
           "The class was a waste of time",
           "I am unhappy with today's session",
           "Today's explanation was totaly unhelpful",
           "I love this class",
           "This lecture is amazing",
           "The teacher explains very well",
           "I enjoyed the session",
           "This subject is interesting",
           "The class was very helpful",
           "I learned a lot today",
           "The explanation was clear",
           "This lesson was easy to understand",
           "The examples were really good",
           "I enjoyed the class"]
Labels=np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])
model=CountVectorizer(lowercase=True,ngram_range=(1,2),stop_words='english')
X=model.fit_transform(Feedbacks)
Classifier=LogisticRegression()
Classifier.fit(X,Labels)
n=input("Enter a feedback: ")
A=model.transform([n])
prediction=Classifier.predict(A)
if prediction[0]==1:
    print("Positive Feedback")
else:
    print("Negative Feedback")
   
