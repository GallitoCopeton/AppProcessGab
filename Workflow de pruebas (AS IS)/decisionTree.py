from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
import sys

import os

def encodeTarget(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target_" + str(target_column)] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

def train(df2, features, target, min_samples_split = 5):
    y = df2[target]
    X = df2[features]
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=99)
    dt.fit(X, y)
    return dt

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        sys.exit("Could not run dot, ie graphviz, to "
             "produce visualization")