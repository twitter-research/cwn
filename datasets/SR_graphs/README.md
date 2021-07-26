This folder contains families of Strongly Regular (SR) graphs.

They represent Regular graphs (each node has the same degree `d`), but with the following additional property:
> Given integers `l`, `u`, every two adjacent nodes have `l` common neighbours and every two non-adjacent nodes have `u` common neighbours.

SR graphs in the same family share the same parameters `n`, `d`, `l`, `u`, with `n` the number of nodes in each graph.
Two non-isomorphic SR graphs in the same family cannot be distinguished by the standard WL test, and not even the more
powerful 3-WL.

In `./raw`, each family is stored in `g6` format and is named as `sr<n><d><l><u>.g6` (two digits for `<n>`). These data
were originally obtained from `http://users.cecs.anu.edu.au/~bdm/data/graphs.html`.
