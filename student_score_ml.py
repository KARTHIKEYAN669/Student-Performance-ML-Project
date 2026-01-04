import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Score": [35, 40, 50, 55, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)
print(df)

plt.scatter(df["Hours_Studied"], df["Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Score")
plt.show()

X = df[["Hours_Studied"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted scores:", predictions)

error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

hours = [[6.5]]
predicted_score = model.predict(hours)
print("Predicted score for 6.5 hours:", predicted_score[0])
