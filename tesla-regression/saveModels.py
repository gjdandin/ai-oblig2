import pickle


def save_model(model, filename):
    """ Save a model to a given file. """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")


def load_model(filename):
    """ Opens a model from a given file."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# save_model(model, 'my_model.sav')
# loaded_model = load_model('my_model.sav')