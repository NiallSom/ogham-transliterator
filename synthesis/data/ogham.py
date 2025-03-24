from dataclasses import dataclass
from enum import Enum, auto



class Direction(Enum):
    HORIZONTAL = auto()
    LEFT = auto()
    RIGHT = auto()
    DIAGONAL = auto()
    P = auto()
    AE = auto()
    IA = auto()
    UI = auto()
    OI = auto()
    EA = auto()
@dataclass
class Ogham:
    lines: int
    direction: Direction

class Characters(Enum):
    N=Ogham(5, Direction.RIGHT)
    S=Ogham(4, Direction.RIGHT)
    F=Ogham(3, Direction.RIGHT)
    L=Ogham(2, Direction.RIGHT)
    B=Ogham(1, Direction.RIGHT)
    Q=Ogham(5, Direction.LEFT)
    C=Ogham(4, Direction.LEFT)
    T=Ogham(3, Direction.LEFT)
    D=Ogham(2, Direction.LEFT)
    H=Ogham(1, Direction.LEFT)
    R=Ogham(5, Direction.DIAGONAL)
    Z=Ogham(4, Direction.DIAGONAL)
    NG=Ogham(3, Direction.DIAGONAL)
    G=Ogham(2, Direction.DIAGONAL)
    M=Ogham(1, Direction.DIAGONAL)
    I=Ogham(5, Direction.HORIZONTAL)
    E=Ogham(4, Direction.HORIZONTAL)
    U=Ogham(3, Direction.HORIZONTAL)
    O=Ogham(2, Direction.HORIZONTAL)
    A=Ogham(1, Direction.HORIZONTAL)
