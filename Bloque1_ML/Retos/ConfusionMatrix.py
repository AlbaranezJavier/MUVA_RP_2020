from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1,4]
y_pred = [2,0,2,2,0,2,4]

cm = confusion_matrix(y_true, y_pred)
print(cm)