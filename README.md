# Ex.No-10-Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:
```
Developed by: Piritharaman R 
RegisterNumber: 212223230148
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
## DATA.HEAD()

![image](https://github.com/user-attachments/assets/51efdcd0-3551-44b4-8ee2-c951b6dbb419)

## DATA.INF0()

![image](https://github.com/user-attachments/assets/5d58543a-762f-4f80-b1b9-5f3f007276ce)

## DATA.ISNULL().SUM()

![image](https://github.com/user-attachments/assets/0a15abe2-b6fc-4b4e-abb8-1b28168b5c6c)

## PLOT USING ELBOW METHOD

![image](https://github.com/user-attachments/assets/b55962be-4c34-4ca4-a62e-0eb7808923f8)

## CUSTOMER SEGMENT

![image](https://github.com/user-attachments/assets/8b97c1e3-4081-4f63-9416-fb9f287356b4)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
