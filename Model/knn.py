from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, Y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    return model