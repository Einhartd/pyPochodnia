import numpy as np

from app.node import Constant
from app.node.operations.arithmetic import Add


def main() -> None:
    c1 = Constant(value=np.array([10, 20, 30], dtype=np.float16))
    print(c1)
    c2 = Constant(value=np.array([1, 2, 3], dtype=np.float16))
    print(c2)

    add1 = Add(a=c1, b=c2, name="Addition_1")
    print(add1)
    add1.forward()
    print("Result of addition:", add1.value)


if __name__ == "__main__":
    main()
