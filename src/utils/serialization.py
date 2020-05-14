import _pickle as cPickle
import pickle


def dump_pkl(df, file_name):
    """
    Dumps a pandas dataframe
    :param df: dataframe
    :param file_name: dumped file name
    :return:
    """
    name = file_name.split("/")[-1]

    with open(file_name, "wb") as file:
        cPickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"{name} dumped")


def load_pkl(file_name):
    name = file_name.split("/")[-1]
    print(f"Loading {name}... ", end="")

    try:
        with open(file_name, "rb") as file_name:
            df = cPickle.load(file_name)
    except FileNotFoundError:
        print("No file found!")
        return None

    print("File loaded")
    return df
