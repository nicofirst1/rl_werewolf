import argparse
import json
import multiprocessing
import os
import shutil
import inspect
import sys
import uuid

import termcolor


def join_paths(path1, path2):
    return os.path.join(path1, path2)


class Params:
    ##########################
    #       Path params
    ##########################

    WORKING_DIR = os.getcwd().split("src")[0]
    SRC_DIR = join_paths(WORKING_DIR, "src")

    LOG_DIR=join_paths(WORKING_DIR,"log_dir")

    RAY_DIR=join_paths(LOG_DIR,"ray_results")
    GAME_LOG_DIR=join_paths(LOG_DIR,"match_log")

    ##########################
    # Performance stuff
    ##########################
    debug = False

    n_cpus = multiprocessing.cpu_count() if not debug else 1
    n_gpus = 1 if not debug else 0

    ##########################
    # env params
    ##########################
    num_player=20

    ##########################
    # other
    ##########################

    unique_id=str(uuid.uuid1())[:8]
    log_match_file=join_paths(GAME_LOG_DIR,f"{unique_id}_log.log")

    
    ##########################
    #    METHODS
    ##########################

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        EXAMPLE_USAGE = "python FlowMas/simulation.py {args}"

        att = self.__get_attributes()

        """Create the parser to capture CLI arguments."""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='[Flow] Evaluates a reinforcement learning agent '
                        'given a checkpoint.',
            epilog=EXAMPLE_USAGE)

        # for every attribute add an arg instance
        for k, v in att.items():
            parser.add_argument(
                "--" + k.lower(), type=type(v), default=v,

            )

        for k, v in vars(parser.parse_args()).items():
            self.__setattr__(k, v)

    def __init__(self):
        print("Params class initialized")
        self.__empty_dirs([self.LOG_DIR])
        self.__initialize_dirs()

        # change values based on argparse
        self.__parse_args()
        self.__log_params()

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def __log_params(self, stdout=sys.stdout):
        """
        Prints attributes as key value on given output
        :param stdout: the output for printing, default stdout
        :return:
        """

         # initializing print message
        hashes=f"\n{20*'#'}\n"
        msg=f"{hashes} PARAMETER START {hashes}"

        # get the attributes ad dict
        attributes = self.__get_attributes()
        # dump using jason
        attributes = json.dumps(attributes, indent=4, sort_keys=True)

        msg+=attributes
        msg+=f"{hashes} PARAMETER END {hashes}"

        color="yellow"
        msg=termcolor.colored(msg,color=color)

        # print them to given out
        print(msg, file=stdout)


    def __initialize_dirs(self):
        """
        Initialize all the directories  listed above
        :return:
        """
        variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for var in variables:
            if var.lower().endswith('dir'):
                path = getattr(self, var)
                if not os.path.exists(path):
                    termcolor.colored(f"Mkdir {path}", "yellow")
                    os.makedirs(path)

    def __empty_dirs(self, to_empty):
        """
        Empty all the dirs in to_empty
        :return:
        """

        for folder in to_empty:
            try:
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
            except Exception:
                continue


