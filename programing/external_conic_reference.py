import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_conic_benchmark import sdpa,cbf,DATA
import numpy as np
from scipy import sparse
import clarabel

def solve(name, p):
 A,b,c,cones=p; A=np.array(A); b=np.array(b); c=np.array(c); n=len(c)
 order=[]; specs=[clarabel.ZeroConeT(len(b))]; off=0
 for typ,dim in cones:
  if typ=='N': order.extend(range(off,off+dim)); specs.append(clarabel.NonnegativeConeT(dim)); off+=dim
  elif typ=='Q': order.extend(range(off,off+dim)); specs.append(clarabel.SecondOrderConeT(dim)); off+=dim
  else:
   # Clarabel's upper-triangle svec is column-major including diagonals.
   for j in range(dim):
    for i in range(j+1): order.append(off+(i if i==j else dim+j*(j-1)//2+i))
   specs.append(clarabel.PSDTriangleConeT(dim)); off+=dim*(dim+1)//2
 C=np.vstack((A,-np.eye(n)[order])); d=np.r_[b,np.zeros(n)]
 settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_iter=200
 solver=clarabel.DefaultSolver(sparse.csc_matrix((n,n)),c,sparse.csc_matrix(C),d,specs,settings)
 r=solver.solve()
 print(name,'status',r.status,'objective',-r.obj_val,'primal',r.r_prim,'dual',r.r_dual,'iterations',r.iterations,flush=True)
if __name__ == '__main__':
 for name in ['truss1','control1','hinf1','truss4','hinf2','hinf3']:
  solve('SDPLIB/'+name,sdpa(DATA/'sdplib'/(name+'.dat-s')))
 for name in ['sambal','gbd','nvs03']:
  solve('CBLIB/'+name+(' (continuous relaxation)' if name != 'sambal' else ''),
        cbf(DATA/'cblib'/(name+'.cbf'),relax_integrality=name != 'sambal'))
