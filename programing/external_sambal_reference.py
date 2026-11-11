"""Clarabel reference on the original CBF primal, including explicit relaxations.

Only the supported free-variable, L=/L+/Q subset is accepted. Integrality is
relaxed only for the two explicitly named CBLIB instances below.
"""
from pathlib import Path
import clarabel
import numpy as np
import scipy.sparse as sp


def solve(name, relax_integrality=False):
    path=Path(__file__).resolve().parent/'testdata/external/cblib'/(name+'.cbf')
    lines=[line.strip() for line in path.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith('#')]
    sections={}; i=0
    while i<len(lines):
        key=lines[i]; i+=1
        if key in ('VER','OBJSENSE'):
            sections[key]=lines[i]; i+=1
        elif key in ('VAR','CON'):
            count,blocks=map(int,lines[i].split()); i+=1
            sections[key]=(count,[lines[i+j].split() for j in range(blocks)]); i+=blocks
        elif key in ('INT','ACOORD','BCOORD','OBJACOORD'):
            if key=='INT' and not relax_integrality:
                raise ValueError('integer model requires explicit relaxation')
            count=int(lines[i]); i+=1
            sections[key]=[lines[i+j].split() for j in range(count)]; i+=count
        else:
            raise ValueError('unsupported CBF section '+key)
    n,variable_cones=sections['VAR']; m,constraint_cones=sections['CON']
    if sections['OBJSENSE']!='MIN' or variable_cones != [['F',str(n)]]:
        raise ValueError('only free-variable minimization models are supported')
    mat=np.zeros((m,n)); h=np.zeros(m); c=np.zeros(n)
    for r,j,v in sections.get('ACOORD',[]): mat[int(r),int(j)]+=float(v)
    for r,v in sections.get('BCOORD',[]): h[int(r)]+=float(v)
    for j,v in sections.get('OBJACOORD',[]): c[int(j)]+=float(v)
    cones=[]
    for kind,size in constraint_cones:
        constructors={'L+':clarabel.NonnegativeConeT,'L=':clarabel.ZeroConeT,'Q':clarabel.SecondOrderConeT}
        if kind not in constructors: raise ValueError('unsupported cone '+kind)
        cones.append(constructors[kind](int(size)))
    settings=clarabel.DefaultSettings(); settings.verbose=False
    result=clarabel.DefaultSolver(sp.csc_matrix((n,n)),c,sp.csc_matrix(-mat),h,cones,settings).solve()
    print(name, 'continuous relaxation' if relax_integrality else 'original continuous problem',
          'status',result.status,'objective',result.obj_val,
          'primal residual',result.r_prim,'dual residual',result.r_dual)


if __name__=='__main__':
    solve('sambal')
    solve('gbd',relax_integrality=True)
    solve('nvs03',relax_integrality=True)
