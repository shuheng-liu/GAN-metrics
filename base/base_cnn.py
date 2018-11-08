from abc import abstractmethod


class BaseModel(object):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_model_vars(self, variable_dict, session):
        pass

    @abstractmethod
    def get_model_vars(self, session, init=False):
        return {}

    @abstractmethod
    def load_model_vars(self, path: str, session):
        pass

    @abstractmethod
    def save_model_vars(self, path: str, session, init=False):
        pass

    @abstractmethod
    def load_model_pretrained(self, session):
        pass

    @abstractmethod
    def _create_loss(self, *args):
        pass
