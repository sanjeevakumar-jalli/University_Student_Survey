from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, Y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)
    return model