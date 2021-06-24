import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas.plotting as pdplt

pd.set_option('display.max_columns', None)

df = pd.read_csv("heart.csv")
df = df.drop_duplicates()

cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

targets = df["target"]
features = df.drop(columns=["ca", "thal", "target"])
# features = pd.get_dummies(features, columns=cat_cols)


# # Matrix plot
# pdplt.scatter_matrix(df, figsize=(10,10))
# plt.show()


# # Univariate Selection
# best_features = SelectKBest(score_func=chi2, k=10)
# fit = best_features.fit(features, targets)
# feature_scores = pd.DataFrame(
#     {
#         "Feature": features.columns.to_list(),
#         "Score": fit.scores_
#     }
# )
# print(feature_scores.nlargest(10, 'Score'))


# # Feature Importance
# model = ExtraTreesClassifier()
# model.fit(features, targets)
# print(model.feature_importances_)
#
# feat_importances = pd.Series(model.feature_importances_, index=features.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()


# Correlation matrix with heatmap
corr_mat = features.corr()
top_corr_features = corr_mat.index
plt.figure(figsize=(20,20))
#plot heat map
g = sns.heatmap(features[top_corr_features].corr(), annot=True)
plt.show()















