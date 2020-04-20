from abc import ABCMeta, abstractmethod


class FooParent(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def static_abstract():
        print("BASE")

    @staticmethod
    def static_inherited():
        FooParent.static_abstract()


class FooChild(FooParent):
    @staticmethod
    def static_abstract():
        print("OVERLOAD")


if __name__ == "__main__":
    FooChild.static_inherited()