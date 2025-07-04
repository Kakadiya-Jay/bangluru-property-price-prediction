import json
import pickle
import numpy as np

# __locations = {}
# __data_columns = {}
# __model = {}


def get_location_name():
    __data_columns, __locations, __model = load_saved_artifacts()
    return __locations


def get_estimated_price(location, sqft, bhk, bath):
    __data_columns, __locations, __model = load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("Loading saved artifacts start....")
    __data_columns = {}
    __locations = {}
    __model = {}

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    # don't write pickle extention here otherwise it will give error.
    with open("./artifacts/banglore_home_prices_model", "rb") as f:
        __model = pickle.load(f)

    print("Loading saved artifacts is done...")

    return __data_columns, __locations, __model


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_name())
    print(get_estimated_price("1st phase jp nagar", 1000, 2, 2))
    print(get_estimated_price("1st phase jp nagar", 1000, 3, 3))
    print(get_estimated_price("Kalhalli", 1000, 2, 2))
    print(get_estimated_price("Ejipura", 1000, 2, 2))
