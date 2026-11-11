#!/usr/bin/env python3
"""Run selected downloaded SDPA/CBF fixtures against the dense Jordan solver.

This intentionally supports only the documented subset in the selected files;
unknown sections and domains raise errors rather than being silently ignored.
Run from the repository root after building //programing:jordan_benchmark_cli.
"""
import math
import os
import argparse
import pathlib
import re
import sys
import subprocess
import time

if os.environ.get('TEST_SRCDIR') and os.environ.get('TEST_WORKSPACE'):
    ROOT = pathlib.Path(os.environ['TEST_SRCDIR']) / os.environ['TEST_WORKSPACE']
    CLI = ROOT / 'programing/jordan_benchmark_cli'
    LP_CLI = ROOT / 'programing/solve_mps'
else:
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    CLI = ROOT / 'bazel-bin/programing/jordan_benchmark_cli'
    LP_CLI = ROOT / 'bazel-bin/programing/solve_mps'
DATA = ROOT / 'programing/testdata/external'


def sdpa(path):
    # SDPA: max F0•Y s.t. Fi•Y=ci, Y>=0. Our solver minimizes -F0•Y.
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith(('*', '"'))]
    m, nb = int(lines[0]), int(lines[1])
    sizes = [int(x) for x in lines[2].replace('{',' ').replace('}',' ').replace(',',' ').split()]
    rhs = [float(x) for x in lines[3].replace('{',' ').replace('}',' ').replace(',',' ').split()]
    if len(sizes) != nb or len(rhs) != m or any(n == 0 for n in sizes):
        raise ValueError('invalid SDPA header')
    cones, offsets, total = [], [], 0
    for n in sizes:
        offsets.append(total)
        if n > 0:
            cones.append(('S', n)); total += n*(n+1)//2
        else:
            cones.append(('N', -n)); total -= n
    A = [[0.0]*total for _ in range(m)]; c = [0.0]*total
    for line in lines[4:]:
        mat, block, row, col, val = line.split()
        mat, block, row, col, val = int(mat), int(block)-1, int(row)-1, int(col)-1, float(val)
        n = sizes[block]
        if not (0 <= mat <= m and 0 <= block < nb and 0 <= row <= col < abs(n)):
            raise ValueError('invalid SDPA coordinate')
        if n < 0:
            if row != col: raise ValueError('off-diagonal in diagonal block')
            index, value = offsets[block]+row, val
        else:
            index = offsets[block]+(row if row == col else n + col*(col-1)//2 + row)
            value = val if row == col else math.sqrt(2)*val
        if mat == 0: c[index] -= value
        else: A[mat-1][index] += value
    return A, rhs, c, cones


def cbf(path, relax_integrality=False):
    # Restricted CBF dual conversion: min c'z, Mz+h in K, z free.
    # The conic dual in this solver's form is min h's, M^T s=c, s in K*.
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith('#')]
    sections = {}; i = 0
    while i < len(lines):
        key=lines[i]; i+=1
        if key in sections: raise ValueError('duplicate '+key)
        if key in ('VER','OBJSENSE'): sections[key]=[lines[i]]; i+=1
        elif key in ('VAR','CON'):
            count, blocks=map(int, lines[i].split()); i+=1
            sections[key]=[count]+[lines[i+j].split() for j in range(blocks)]; i+=blocks
        elif key == 'INT':
            if not relax_integrality: raise ValueError('integer CBF requires explicit continuous relaxation')
            count=int(lines[i]); i+=1
            sections[key]=[int(lines[i+j]) for j in range(count)]; i+=count
        elif key in ('OBJACOORD','ACOORD','BCOORD'):
            count=int(lines[i]); i+=1
            sections[key]=[lines[i+j].split() for j in range(count)]; i+=count
        else: raise ValueError('unsupported CBF section '+key)
    if sections['OBJSENSE'] != ['MIN'] or sections['VAR'][1:] != [['F',str(sections['VAR'][0])]]:
        raise ValueError('only minimization with free variables supported')
    nv, nr=sections['VAR'][0], sections['CON'][0]
    if any(index < 0 or index >= nv for index in sections.get('INT', [])):
        raise ValueError('integer variable index out of range')
    if nv * nr > 2_000_000: raise ValueError('CBF model exceeds dense converter limit')
    M=[[0.0]*nv for _ in range(nr)]; h=[0.0]*nr; obj=[0.0]*nv
    for row,col,val in sections.get('ACOORD',[]): M[int(row)][int(col)]+=float(val)
    for row,val in sections.get('BCOORD',[]): h[int(row)]+=float(val)
    for col,val in sections.get('OBJACOORD',[]): obj[int(col)]+=float(val)
    cones=[]; columns=[]; costs=[]; offset=0
    for domain, dim in sections['CON'][1:]:
        dim=int(dim)
        if domain == 'L=':
            # Its dual is free: represent it as positive minus negative.
            for row in range(offset,offset+dim):
                cones.extend([('N',1),('N',1)]); columns.extend([(row,1),(row,-1)])
        elif domain in ('L+','Q'):
            cones.append(('N' if domain == 'L+' else 'Q',dim))
            columns.extend((row,1) for row in range(offset,offset+dim))
        else: raise ValueError('unsupported CBF cone '+domain)
        offset+=dim
    if offset != nr: raise ValueError('CBF cone dimensions disagree')
    A=[[M[row][j]*sign for row,sign in columns] for j in range(nv)]
    costs=[h[row]*sign for row,sign in columns]
    return A,obj,costs,cones


def run(name, data, target=None):
    A,b,c,cones=data; m,n=len(b),len(c)
    rows=[f'{m} {n} {len(cones)}']+[f'{t} {s}' for t,s in cones]
    rows += [' '.join(map(str,row)) for row in A]
    rows += [' '.join(map(str,b)), ' '.join(map(str,c))]
    start=time.monotonic()
    result=subprocess.run([str(CLI)],input='\n'.join(rows)+'\n',capture_output=True,text=True,timeout=120,check=True)
    elapsed=time.monotonic()-start
    out=result.stdout.splitlines(); status,it,pr,du,gap=out[0].split()
    x=[float(z) for z in out[1].split()]
    value=-sum(a*b for a,b in zip(c,x)) # primal maximization / original minimization
    print(f'{name}: status={status} iterations={it} target={target} objective={value:.10g} '
          f'primal_residual={pr} dual_residual={du} gap={gap} elapsed_s={elapsed:.3f}',flush=True)
    return int(status), value, x, [float(z) for z in out[2].split()], [float(z) for z in out[3].split()]


def check_cones(vector, cones):
    # Independent cone check; a small diagonal shift permits roundoff at PSD boundaries.
    off=0
    for kind, size in cones:
        if kind == 'N':
            assert min(vector[off:off+size]) >= -1e-7
            off += size
        elif kind == 'Q':
            v=vector[off:off+size]
            assert v[0] + 1e-7 >= math.hypot(*v[1:])
            off += size
        else:
            dim=size*(size+1)//2
            v=vector[off:off+dim]; M=[[0.0]*size for _ in range(size)]
            for i in range(size): M[i][i]=v[i]
            k=size
            for j in range(1,size):
                for i in range(j):
                    M[i][j]=M[j][i]=v[k]/math.sqrt(2); k+=1
            scale=max(1.0,max(abs(M[i][i]) for i in range(size)))
            # Cholesky of M + tolerance*I tests approximate positive semidefiniteness.
            L=[[0.0]*size for _ in range(size)]
            for i in range(size):
                for j in range(i+1):
                    value=M[i][j]+(1e-12*scale if i==j else 0)-sum(L[i][k]*L[j][k] for k in range(j))
                    if i==j:
                        assert value > 0, 'PSD block has a negative eigenvalue'
                        L[i][j]=math.sqrt(value)
                    else: L[i][j]=value/L[j][j]
            off+=dim
    assert off==len(vector)


def verify(name, data, target, result):
    A,b,c,cones=data; status,value,x,y,s=result
    assert status==0, f'{name}: solver did not converge (status {status})'
    assert abs(value-target) <= 1e-5*(1+abs(target)), f'{name}: objective mismatch'
    assert len(x)==len(c) and len(y)==len(b) and len(s)==len(c)
    rp=math.sqrt(sum((sum(a*z for a,z in zip(row,x))-bi)**2 for row,bi in zip(A,b)))
    rd=math.sqrt(sum((c[j]-sum(A[i][j]*y[i] for i in range(len(b)))-s[j])**2 for j in range(len(c))))
    gap=sum(a*z for a,z in zip(x,s))
    assert rp <= 1e-8*(1+math.sqrt(sum(z*z for z in b)))
    assert rd <= 1e-8*(1+math.sqrt(sum(z*z for z in c)))
    assert -1e-10 <= gap <= 1e-8*(1+abs(sum(a*z for a,z in zip(c,x))))
    check_cones(x,cones); check_cones(s,cones)
    print(name, 'independent objective, KKT and cone checks passed', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='fail if any benchmark or KKT check fails')
    parser.add_argument('--extended', action='store_true', help='include difficult cases that currently do not converge')
    args=parser.parse_args()
    failures=[]
    print('Netlib LP results (unified Jordan sparse LP backend):', flush=True)
    lp_cases=[('sc50a', -64.5750770586), ('sc50b', -70.0),
              ('blend', -30.812149846), ('sc105', -52.202061212)]
    if args.extended:
        lp_cases += [('sc205', -52.202061212), ('recipe', -266.616)]
    for name,target in lp_cases:
        p=subprocess.run([str(LP_CLI),str(DATA/'netlib'/(name+'.mps'))],
                         capture_output=True,text=True,timeout=120,check=False)
        relevant=[line for line in p.stdout.splitlines() if line.startswith(
            ('Standard form rows:', 'Status:', 'Elapsed:', 'Objective value:', 'Primal residual', 'Dual residual', 'Complementarity gap'))]
        print('Netlib/'+name, 'target='+str(target), ' | '.join(relevant),flush=True)
        objective=re.search(r'^Objective value: ([^\n]+)',p.stdout,re.MULTILINE)
        if p.returncode != 0 or not objective or abs(float(objective.group(1))-target)>1e-5*(1+abs(target)):
            failures.append('Netlib/'+name+': status or objective check failed')
    print('Jordan status codes: 0=optimal, 1=iteration limit, 2=numerical failure, 3=invalid input',flush=True)
    sdp_cases=[('truss1',-8.999996),('control1',17.78463),('hinf1',2.0326),('truss4',-9.009996)]
    if args.extended:
        # hinf2 uses SDPLIB's rounded reference; Clarabel's AlmostSolved
        # value differs and is not an exact reference. hinf3 uses that
        # approximate independent value as a cross-check.
        sdp_cases += [('hinf2',10.967),('hinf3',56.94094238)]
    for name,target in sdp_cases:
        data=sdpa(DATA/'sdplib'/(name+'.dat-s'))
        result=run('SDPLIB/'+name,data,target)
        if args.verify:
            try: verify(name,data,target,result)
            except AssertionError as error: failures.append('SDPLIB/'+name+': '+str(error))
    for name,target,relax in [('sambal',3.96822061,False),('gbd',2.2,True),('nvs03',8.152139962,True)]:
        path=DATA/'cblib'/(name+'.cbf')
        if args.verify and relax:
            try:
                cbf(path)
            except ValueError as error:
                assert 'explicit continuous relaxation' in str(error)
            else:
                raise AssertionError('integer CBF unexpectedly accepted without relaxation')
        data=cbf(path,relax_integrality=relax)
        label='CBLIB/'+name+(' (continuous relaxation)' if relax else '')
        result=run(label,data,target)
        if args.verify:
            try: verify(label,data,target,result)
            except AssertionError as error: failures.append(label+': '+str(error))
    if failures:
        print('Benchmark failures:', *failures, sep='\n  ', flush=True)
        if args.verify: sys.exit(1)
