# import joblib
# from sklearn.tree import export_text

# model = joblib.load("decision_tree_model.pkl")

# print(type(model))
# print("Dybde:", model.get_depth())
# print("Blade:", model.get_n_leaves())

# print(export_text(model))

import joblib
import matplotlib.pyplot as plt
from sklearn import tree

model = joblib.load("decision_tree_model.pkl")

plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True)
plt.show()