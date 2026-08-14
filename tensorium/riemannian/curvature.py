from tensorium.core.chart import chart_diff
from itertools import product
from sympy import factor_terms, cancel
from tensorium.fields.local_tensor_field import LocalTensorField
from tensorium.fields.tensor_field import TensorField
from tensorium.riemannian.metric import CovariantMetricTensor, LocalCovariantMetricTensor
from sympy.core import S

def riemann_from_affine_conection(affine_connection):
    if not hasattr(affine_connection,"local_representations"):
        raise ValueError("affine_connection must be an AffineConnection")
    
    dim=affine_connection.dim
    local_representations={}

    for chart, local_conn in affine_connection.local_representations.items():
        conn_get=local_conn.__getitem__
        diff_cache={}
        product_cache={}

        for rho,sigma,mu,nu in product(range(dim),repeat=4):
            diff_cache[(rho,sigma,mu,nu)]=chart_diff(conn_get((rho,nu,sigma)), chart, mu)
            product_cache[(rho,sigma,mu,nu)]=sum(conn_get((rho,mu,lam))*conn_get((lam,nu,sigma))
                for lam in range(dim))
        def diff_cache_get(rho,sigma,mu,nu): return diff_cache[(rho,sigma,mu,nu)]
        def product_cache_get(rho,sigma,mu,nu): return product_cache[(rho,sigma,mu,nu)]

        data=[]
        for rho,sigma in product(range(dim),repeat=2):
            for mu in range(dim):
                for nu in range(mu,dim):
                    if mu==nu: data.append(S.Zero)
                    else: data.append(factor_terms(cancel(diff_cache_get(rho,sigma,mu,nu)
                            -diff_cache_get(rho,sigma,nu,mu)+product_cache_get(rho,sigma,mu,nu)
                            -product_cache_get(rho,sigma,nu,mu))))
        equivalences={(rho,sigma,mu,nu): ((rho,sigma,nu,mu),-1) for rho in range(dim) for sigma in range(dim)
            for mu in range(dim) for nu in range(dim) if mu > nu}

        local_representations[chart]=LocalTensorField(chart,(1, 3),tuple(data),(1,-1,-1,-1),equivalences)

    return TensorField(affine_connection.manifold,(1,3),local_representations,(1,-1,-1,-1))


riemann_from_affine_connection = riemann_from_affine_conection


def local_ricci_tensor_from_metric(local_metric, backend="sympy", simplify=True, simplify_backend=None, strategy="fast", return_timings=False):
    """Compute the local Ricci tensor from a local metric."""
    if not isinstance(local_metric, LocalCovariantMetricTensor):
        raise ValueError("local_metric must be a LocalCovariantMetricTensor")
    chart=local_metric.chart
    dim=chart.dim
    g_cov=local_metric
    g_contra=local_metric.inverse()
    dg={(a,b,c): chart_diff(g_cov[(a,b)],chart,c) for a in range(dim) for b in range(a,dim) for c in range(dim)}
    def dg_get(a,b,c):
        if a<=b: return dg[(a,b,c)]
        return dg[(b,a,c)]
    gamma={}
    for i,j,k in product(range(dim),repeat=3):
        total=S.Zero
        for l in range(dim):
            total+=g_contra[(i,l)]*(dg_get(l,j,k)+dg_get(l,k,j)-dg_get(j,k,l))
        gamma[(i,j,k)]=factor_terms(cancel(S.Half*total)) if simplify else S.Half*total
    def conn_get(indices): return gamma[indices]
    data=[]
    for sigma,nu in product(range(dim),repeat=2):
        total=S.Zero
        for rho in range(dim):
            term=chart_diff(conn_get((rho,nu,sigma)),chart,rho)-chart_diff(conn_get((rho,rho,sigma)),chart,nu)
            term+=sum(conn_get((rho,rho,lam))*conn_get((lam,nu,sigma))
                -conn_get((rho,nu,lam))*conn_get((lam,rho,sigma)) for lam in range(dim))
            total+=term
        data.append(factor_terms(cancel(total)) if simplify else total)
    ricci=LocalCovariantMetricTensor(chart,tuple(data[i*dim+j] for i in range(dim) for j in range(i,dim)))
    if return_timings: return ricci, {}
    return ricci


def ricci_tensor_from_metric(metric, backend="sympy", simplify=True, simplify_backend=None, strategy="fast", return_timings=False):
    """Compute the Ricci tensor of a covariant metric tensor field."""
    if not isinstance(metric, CovariantMetricTensor): raise ValueError("metric must be a CovariantMetricTensor")
    local_representations={}
    timings={}
    for chart,local_metric in metric.local_representations.items():
        if return_timings:
            local_representations[chart],timings[chart]=local_ricci_tensor_from_metric(local_metric,backend=backend,
                simplify=simplify,simplify_backend=simplify_backend,strategy=strategy,return_timings=True)
        else:
            local_representations[chart]=local_ricci_tensor_from_metric(local_metric,backend=backend,
                simplify=simplify,simplify_backend=simplify_backend,strategy=strategy)
    result=TensorField(metric.manifold,(0,2),local_representations,(-1,-1))
    if return_timings: return result,timings
    return result
