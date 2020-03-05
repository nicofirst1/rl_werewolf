import os


def get_file_lines(file_path):
    """
    get the number of lines in a file
    :param file_path: the path to the file
    :return: int, number of lines
    """

    lines = 0

    with open(file_path, "r+") as file:
        for line in file:
            lines += 1

    return lines


def remove_lines_from_file(file_path, line_indices):
    """
    Remove some lines from a file
    :param file_path: the file path
    :param line_indices: a list of line indices to be removed
    :return:
    """

    lines = []
    idx = 0

    print(f"removing lines: {line_indices}")

    with open(file_path, "r+") as file:

        for line in file:
            if idx not in line_indices:
                lines.append(line)
            idx += 1

    with open(file_path, "w+") as file:
        file.writelines(lines)


def empty_dir(log_dir_path):
    print(f"Emptying {log_dir_path}")
    for file in os.listdir(log_dir_path):
        os.remove(os.path.join(log_dir_path, file))


def get_number_of_files(path2dir):
    """
    return the number of files in a direcotry
    :param path2dir: the path to the dir
    :return:
    """
    return len([name for name in os.listdir(path2dir) if os.path.isfile(os.path.join(path2dir, name))])


def get_files_with_name(path2dir, name):
    """
    Return a list of file's paths in the dirpath2dir which have name in them
    :param path2dir: the dir to look into
    :param name: a sub name to match
    :return: list of paths
    """
    return [os.path.join(path2dir, f) for f in os.listdir(path2dir) if name in os.path.join(path2dir, f)]


def check_create_dir(dir_name):
    """
    Create direcotry if not exists
    :param dir_name: (str) the name of the dir
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
