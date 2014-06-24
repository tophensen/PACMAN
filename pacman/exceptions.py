__author__ = 'daviess,stokesa6'


class PacmanException(Exception):
    """
    Superclass of all exceptions from the pacman module.

    :raise None: does not raise any known exceptions
    """
    pass


class InvalidChipException(PacmanException):
    """
    Exception thrown when a subvertex is placed in a chip\
    which does not exist in the machine structure

    :raise None: does not raise any known exceptions
    """
    pass


class InvalidProcessorException(PacmanException):
    """
    Exception thrown when a subvertex is placed in a processor\
    which does not exist in the machine structure

    :raise None: does not raise any known exceptions
    """
    pass


class ChipAlreadyExistsException(PacmanException):
    """
    Exception thrown when a chip that is being added already\
    exists in the machine

    :raise None: does not raise any known exceptions"""
    pass

class ProcessorAlreadyExistsException(PacmanException):
    """
    Exception thrown when a processor that is being added already\
    exists in the machine

    :raise None: does not raise any known exceptions"""
    pass

class ProcessorAlreadyInUseException(PacmanException):
    """
    Exception thrown when a subvertex is mapped to a processor\
    which is already target of the placement for a different subvertex

    :raise None: does not raise any known exceptions"""
    pass


#class Exception(PacmanException):
#    """thrown when a response code from the spinnaker board
#       is not recongised by spinnman
#    :raise None: does not raise any known exceptions"""
#    pass