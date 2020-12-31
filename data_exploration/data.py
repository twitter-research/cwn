class Chain(object):
    '''
        Class representing a chain of k-order simplices.
    '''
    def __init__(self, order, x=None, upper_index=None, lower_index=None, y=None, mapping=None):
        '''
            Constructs a `order`-chain.
            - `order`: order of the simplices in the chain
            - `x`: feature matrix, shape [num_simplices, num_features]; may not be available
            - `upper_index`: upper adjacency, matrix, shape [2, num_upper_connections]; may not be available, e.g. when `order` is the top level order of a complex
            - `lower_index`: lower adjacency, matrix, shape [2, num_lower_connections]; may not be available, e.g. when `order` is the top level order of a complex
            - `y`: labels over simplices in the chain, shape [num_simplices,]
            - `mapping`: dictionary, simplex_id to nodes
        '''
        self.order = order
        self.x = x
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.y = y
        self.mapping = mapping
        self.oriented = False
        self._hodge_laplacian = None
        return

    def orient(self, arbitrary=None):
        '''
            Enforces orientation to the chain.
            If `arbitrary` orientation is provided, it enforces that. Otherwise the canonical one is enforced.
        '''
        raise NotImplementedError
        # TODO: what is the impact of this on lower/upper signals?
        # ...
        # ...
        # self.lower_orientation = ...  # <- shape [1, num_lower_connections], content: +/- 1.0
        # self.upper_orientation = ...  # <- shape [1, num_upper_connections], content: +/- 1.0
        # self.oriented = True
        # return

    def get_hodge_laplacian(self):
        '''
            Returns the Hodge Laplacian.
            Orientation is required; if not present, the chain will first be oriented according to the canonical ordering.
        '''
        raise NotImplementedError
        # if self._hodge_laplacian is None:  # caching
        #     if not self.oriented:
        #         self.orient()
        #     self._hodge_laplacian = ...
        #     # ^^^ here we need to perform two sparse matrix multiplications -- we can leverage on torch_sparse
        #     # indices of the sparse matrices are self.lower_index and self.upper_index, their values are those in 
        #     # self.lower_orientation and self.upper_orientation
        # return self._hodge_laplacian

    def initialize_features(self, strategy='constant'):
        '''
            Routine to initialize simplex-wise features based on the provided `strategy`.
        '''
        raise NotImplementedError
        # self.x = ...
        # return


class Complex(object):
    '''
        Class representing an attributed simplicial complex.
    '''

    def __init__(self, nodes, edges, triangles, y=None):
        
        self.nodes = nodes
        self.edges = edges
        self.triangles = triangles
        self.chains = {0: self.nodes, 1: self.edges, 2: self.triangles}
        self.min_order = 0
        self.max_order = 2
        # ^^^ these will be abstracted later allowing for generic orders;
        # in that case they will be provided as an array (or similar) and min and
        # max orders will be set automatically, according to what passed
        # NB: nodes, edges, triangles are objects of class `Order` with orders 0, 1, 2
        self.y = y  # complex-wise label for complex-level tasks
        return

    def get_inputs(self, order):
        '''
            Conveniently returns all necessary input parameters to perform higher-order neural message passing at the specified `order`.
        '''
        if order in self.orders:
            simplices = self.chains[order]
            x = simplices.x
            if order < self.max_order:
                upper_index = simplices.upper_index
                upper_features = self.chains[order + 1].x
            else:
                upper_index = None
                upper_features = None
            if order > self.min_order:
                lower_index = simplices.lower_index
                lower_features = self.chains[order - 1].x
            else:
                lower_index = None
                lower_features = None
            inputs = (x, upper_index, upper_features, lower_index, lower_features)
        else:
            raise NotImplementedError('Order {} is not present in the complex or not yet supported.'.format(order))
        return inputs

    def get_labels(self, order=None):
        '''
            Returns target labels.
            If `order`==k (integer in [self.min_order, self.max_order]) then the labels over k-simplices are returned.
            In the case `order` is None the complex-wise label is returned.
        '''
        if order is None:
            y = self.y
        else:
            if order in self.chains:
                y = self.chains[order].y
            else:
                raise NotImplementedError('Order {} is not present in the complex or not yet supported.'.format(order))
        return y

