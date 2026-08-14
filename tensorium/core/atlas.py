from sympy.core import Tuple
from sympy.matrices import ImmutableDenseMatrix

class Atlas:
    """Represent an atlas for a manifold.
    An atlas stores a collection of charts together with caches for
    transition paths, composed transformations, and Jacobian matrices.
    It allows transformations to be computed even when they require several
    intermediate charts.
    """
    def __init__(self, manifold, charts=None):
        """Create an atlas for a manifold.
        Parameters:
         - manifold : Manifold (Manifold described by the atlas)
         - charts : iterable of Chart, optional
            Initial charts to register in the atlas.
        """
        self._manifold=manifold
        self._charts=[]
        self._path_cache={}
        self._transform_cache={}
        self._jacobian_cache={}
        if charts is not None:
            for chart in charts:
                self.add_chart(chart)
    
    @property
    def manifold(self):
        """Return the manifold described by the atlas."""
        return self._manifold
    @property
    def charts(self):
        """Return the registered charts."""
        return self._charts
    @property
    def path_cache(self):
        """Return the cache of chart paths."""
        return self._path_cache
    @property
    def transform_cache(self):
        """Return the cache of composed chart transformations."""
        return self._transform_cache
    @property
    def jacobian_cache(self):
        """Return the cache of Jacobian matrices."""
        return self._jacobian_cache

    def add_chart(self, chart):
        """Add a chart to the atlas. Parameters:
         - chart : Chart (Chart to register)
        """
        if chart.manifold!=self.manifold:
            raise ValueError("The chart does not belong to this manifold")
        self.charts.append(chart)
    
    def neighbors(self, chart):
        """Return charts directly connected to a given chart.
        Two charts are considered neighbors when at least one transition
        relation is known between them.
        Parameters:
         - chart : Chart (Chart whose neighbors are requested)
        Returns:
         - list of Chart: Charts directly connected to ``chart``.
        """
        neigh=[]
        for i in self.charts:
            if i==chart: continue
            else:
                if i in chart.relations or chart in i.relations:
                    neigh.append(i)
        return neigh
    
    def find_path(self, start, end):
        """Find a path of compatible chart transitions between two charts.
        Parameters:
         - start : Chart (Starting chart)
         - end : Chart (Target chart)
        Returns:
         - list of Chart: Sequence of charts connecting ``start`` to ``end``.
        Notes:
        A breadth-first search is used, and successful paths are cached in both
        directions.
        """
        key=(start, end)
        key_inverse=(end, start)
        if key in self.path_cache: return self.path_cache[key]
        if start==end:
            self.path_cache[key]=[start]
            self.path_cache[key_inverse]=[end]
            return [start]
        visited={start}
        queue=[(start,[start])]
        while queue:
            current, path=queue.pop(0)
            for neigh in self.neighbors(current):
                if neigh==end:
                    path=path+[neigh]
                    self.path_cache[key]=path
                    self.path_cache[key_inverse]=list(reversed(path))
                    return path
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, path+[neigh]))
        raise KeyError("The two charts are not connected in the atlas.")
    
    def transform(self, start, end):
        """Compose chart transitions from one chart to another.
        Parameters:
         - start : Chart (Chart in which the input coordinates are expressed)
         - end : Chart (Chart to which the coordinates should be transformed)
        Returns:
         - Tuple: Coordinates of ``end`` written as functions of the coordinates of
            ``start``.
        Notes:
        If no direct transformation is known, the atlas composes the
        transformations along a path of neighboring charts and caches the
        result.
        """
        key=(start,end)
        if key in self.transform_cache: return self.transform_cache[key]
        if start==end:
            res= Tuple(*start.symbols)
            self.transform_cache[key]=res
            start._transform_cache[end]=res
            return res
        path=self.find_path(start, end)
        exprs=start._direct_transform(path[1])

        for i in range(1,len(path)-1):
            subs_rule=dict(zip(path[i].symbols, exprs))
            exprs=Tuple(*[expr.xreplace(subs_rule) for expr in path[i]._direct_transform(path[i+1])])
        self.transform_cache[key]=exprs
        start._transform_cache[end]=exprs
        return exprs
    
    def jacobian(self, start, end):
        """Return the Jacobian of the atlas transformation between two charts.
        Parameters:
         - start : Chart (Source chart)
         - end : Chart (Target chart)
        Returns:
         - ImmutableDenseMatrix:
                Jacobian matrix of the transformation from ``start`` to ``end``.
        Notes:
        The result is cached. When possible, the inverse Jacobian is also cached
        for the reverse chart order.
        """
        key=(start,end)
        if key in self.jacobian_cache: return self.jacobian_cache[key]
        result=ImmutableDenseMatrix(self.transform(start, end)).jacobian(start.symbols)
        self._jacobian_cache[key]=result
        inverse_key=(end,start)
        if not inverse_key in self.jacobian_cache:
            subs_rule=dict(zip(start.symbols, end.transform(start)))
            self.jacobian_cache[inverse_key]=result.inv().xreplace(subs_rule)
        return result
    
