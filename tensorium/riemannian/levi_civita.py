from sympy import cancel, factor_terms, trigsimp, powsimp, sinh, cosh, sin, cos, S
from tensorium.connections.base import AffineConnection, LocalAffineConnection
from tensorium.core.manifold import MetricManifold
from tensorium.core.chart import Chart, chart_diff
from itertools import product


class LeviCivitaConnection(AffineConnection):
    def __new__(cls, manifold):
        if not isinstance(manifold, MetricManifold):
            raise ValueError("The connection can only be defined on a MetricManifold.")
        data={chart: LocalLeviCivitaConnection.from_metric_manifold(manifold, chart)
            for chart in manifold.covariant_metric.local_representations}
        obj=super().__new__(cls, manifold, data)
        return obj
    
    def _best_source_chart(self, target_chart):
        """Select the known source chart that is closest to a target chart.
        Parameters:
         - target_chart : Chart
            Chart in which a local representation is requested.
        Returns:
         - Chart or None:
            Known chart from which the transformation to ``target_chart`` uses
            the shortest atlas path, or ``None`` if no suitable chart is found.
        """
        best_chart=None
        best_length=None
        for chart in self.local_representations:
            path=target_chart.manifold.atlas.find_path(chart, target_chart)
            if best_length is None or len(path)<best_length:
                best_length=len(path)
                best_chart=chart
        return best_chart
    
    def local_representation(self, target_chart):
        if target_chart in self.local_representations:
            return self.local_representations[target_chart]
        local_conn=LocalLeviCivitaConnection.from_metric_manifold(self.manifold, target_chart)
        self._local_representations[target_chart]=local_conn
        return local_conn


class LocalLeviCivitaConnection(LocalAffineConnection):
    def __new__(cls, chart, components):
        if not isinstance(chart, Chart):
            raise ValueError("The local connection must be defined on a chart.")
        if not isinstance(components, tuple):
            raise ValueError("The components of the local connection must be a tuple.")
        if len(components)!=(chart.dim**2)*(chart.dim+1)//2:
            raise ValueError("Given the symmetries of the Levi-Civita connection, " \
            "the components of the local connection must be a tuple of length n^2*(n+1)/2, " \
            "where n is the dimension of the chart.")
        obj=super().__new__(cls, chart, components)
        obj._chart=chart
        obj._components=components
        obj._equivalences={ (i,j,k):((i,k,j)) for i in range(chart.dim) for j in range(chart.dim) for k in range(chart.dim) if j>k}
        canonical_indices=[(i,j,k) for i in range(chart.dim) for j in range(chart.dim) for k in range(j,chart.dim)]
        obj._canonical_positions={idx: pos for pos, idx in enumerate(canonical_indices)}
        return obj
    
    @classmethod
    def from_metric_manifold(cls, metric_manifold, chart):
        if not isinstance(metric_manifold, MetricManifold):
            raise ValueError("The connection can only be defined on a MetricManifold.")
        g_cov=metric_manifold.covariant_metric.local_representation(chart)
        g_contra=metric_manifold.contravariant_metric.local_representation(chart)
        g_cov_get=g_cov.__getitem__
        g_contra_get=g_contra.__getitem__
        dim=chart.dim
        components=[]
        dg={(a,b,c): chart_diff(g_cov_get((a,b)), chart, c)
            for a in range(dim) for b in range(a,dim) for c in range(dim)}
        def dg_get(a,b,c):
            if a<=b: return dg[(a,b,c)]
            return dg[(b,a,c)]

        for i in range(dim):
            for j in range(dim):
                for k in range(j,dim):
                    coeff=S.Zero
                    for l in range(dim):
                        coeff+=g_contra_get((i,l))*(dg_get(l,j,k)+dg_get(l,k,j)-dg_get(j,k,l))
                    coeff*=S.Half
                    coeff=coeff.rewrite(sinh).rewrite(cosh)
                    coeff=coeff.rewrite(sin).rewrite(cos)
                    coeff=powsimp(coeff,force=True)
                    coeff=trigsimp(coeff)
                    coeff=factor_terms(cancel(coeff))
                    components.append(coeff)
        return cls(chart, tuple(components))
    
    def __getitem__(self, indices):
        if len(indices)!=3:
            raise ValueError("The local connection must be indexed with 3 indices.")
        i,j,k=indices
        dim=self.chart.dim
        if not (0<=i<dim and 0<=j<self.chart.dim and 0<=k<dim): raise ValueError("Some index is out of range.")
        if indices in self._equivalences:
            canonical_index=self._equivalences.get(indices, indices)
        else: canonical_index=indices
        pos=self._canonical_positions[canonical_index]
        return self.components[pos]
    @property
    def chart(self): return self.args[0]
    @property
    def components(self): return self.args[1]
    @property
    def equivalences(self): return self._equivalences
    @property
    def canonical_positions(self): return self._canonical_positions

    def transform_to(self, new_chart):
        if new_chart==self.chart: return self
        if new_chart.dim!=self.chart.dim:
            raise ValueError("Dimensions of the old coordinate system and the new coordinate systems do not coincide")
        
        dim=self.chart.dim

        J=self.chart.jacobian(new_chart)
        Jinv=new_chart.jacobian(self.chart)
        
        old_as_functions_of_new=new_chart.transform(self.chart)
        subs_rule=list(zip(self.chart.symbols, old_as_functions_of_new))
        subs_rule+=list(zip(self.chart.args[2], old_as_functions_of_new))
        connection_get=self.__getitem__

        H={(sigma,mu,nu): chart_diff(Jinv[(sigma, mu)], new_chart, nu)
            for sigma in range(dim) for mu in range(dim) for nu in range(mu, dim)}
        def H_get(sigma, mu, nu):
            if mu<=nu:return H[(sigma,mu,nu)]
            return H[(sigma,nu,mu)]
        
        J_sub={(i,j):J[i,j].subs(subs_rule) for i in range(dim) for j in range(dim)}
        all_indices=tuple(product(range(dim), repeat=3))
        target_indices=sorted(index for index in all_indices if index not in self.equivalences)
        connection_sub={idx: connection_get(idx).subs(subs_rule) for idx in all_indices}
        data=[]
        for multi_index in target_indices:
            rho_p,mu_p,nu_p = multi_index
            tensorial=S.Zero
            for sigma, alpha, beta in product(range(dim), repeat=3):
                tensorial+=(J_sub[(rho_p, sigma)]*Jinv[(alpha, mu_p)]
                            *Jinv[(beta, nu_p)]*connection_sub[(sigma,alpha,beta)])
            inhom = S.Zero
            for sigma in range(dim):
                inhom+=J_sub[(rho_p, sigma)]*H_get(sigma, mu_p, nu_p)
            data.append(factor_terms(cancel(tensorial+inhom)))
        components=tuple(data)
        return LocalLeviCivitaConnection(new_chart,components)
