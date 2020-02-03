
class Episode:
    """
    Class to hold info about a training episode, that is from the start of a game till one wins
    """

    def __init__(self):
        self.days = 0
        self.targets = []

    def add_target(self, target_mat):
        """
        Add a target to the target list and increment day
        :param target_mat: nxn matrix, the target output
        :return: None
        """
        self.targets.append(target_mat)
        self.days += 1
