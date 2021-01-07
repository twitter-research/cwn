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

    def inspect(self, func: Callable,
                pop_first: bool = False,
                pop_first_two: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        elif pop_first_two:
            params.popitem(last=False)
            params.popitem(last=False)
        self.params[func.__name__] = params
