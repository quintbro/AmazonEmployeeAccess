import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier

data = pd.read_csv("train.csv", delimiter=",")
test = pd.read_csv("test.csv")

total = data.shape[0]

for column in data.columns[1:]:
    others = data[column].value_counts() / total > 0.001
    categories_to_keep = others[others].index.tolist()
    data[column] = data[column].where(data[column].isin(categories_to_keep), 'Other')

total = test.shape[0]

for column in test.columns[1:]:
    others = test[column].value_counts() / total > 0.001
    categories_to_keep = others[others].index.tolist()
    test[column] = test[column].where(test[column].isin(categories_to_keep), 'Other')

encoder = TargetEncoder(cols = list(data[:0])[1:])
encoder.fit(data, data["ACTION"])
train = encoder.transform(data)
test = encoder.transform(test)

y = train.ACTION.to_numpy()
X = train.drop("ACTION", axis=1).to_numpy()

model = RandomForestClassifier()

param_grid = {
    "n_jobs" : [10],
    "n_estimators" : [1000],
    "max_features" : ['sqrt'],
    "min_samples_split" : [4]
}

grid_search = GridSearchCV(model, param_grid, cv = 10, scoring = "roc_auc")

grid_search.fit(X, y)

best_rf = grid_search.best_estimator_

Knn_mod = KNeighborsClassifier()

param_grid = {
    "algorithm" : ["ball_tree"],
    "weights" : ["uniform"],
    "n_jobs" : [10],
    "n_neighbors" : [8]
}

grid_search_knn = GridSearchCV(Knn_mod, param_grid, cv = 10, scoring = "roc_auc")

grid_search_knn.fit(X, y)

best_knn = grid_search_knn.best_estimator_

stack_mod = StackingClassifier([('rf', best_rf), ('knn', best_knn)], stack_method = 'predict_proba', n_jobs = 10, cv = 'prefit')

stack = stack_mod.fit(X, y)

preds = stack.predict_proba(test.drop("id", axis = 1).to_numpy())[:,1]

submission = pd.DataFrame({"Id" : test.id, "Action" : preds})

submission.to_csv("pysubmission.csv", sep = ",", index = False)