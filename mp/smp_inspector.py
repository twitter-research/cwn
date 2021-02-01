import re
import inspect
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable, Set
from torch_geometric.nn.conv.utils.inspector import Inspector


class SimplicialInspector(Inspector):

    def __implements__(self, cls, func_name: str) -> bool:
        if cls.__name__ == 'ChainMessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def inspect(self, func: Callable, pop_first_n: int = 0) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        for _ in range(pop_first_n):
            params.popitem(last=False)
        self.params[func.__name__] = params
