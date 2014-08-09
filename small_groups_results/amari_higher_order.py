import numpy as np
from itertools import combinations
from pyentropy import maxent
from numpy.random import choice
from IPython import embed
import matplotlib.pyplot as plt
from numpy.random import rand
from sympy import S,solve,sympify,nsolve,lambdify
from itertools import chain, combinations

def get_indices_position_maps(dim):
    i2p = {}
    p2i = {}
    base = [0 for i in range(dim)]
    for pos in xrange(2**dim):
        idcs = np.copy((base))
        bin_nr = bin(pos)[2:]
        for i in range(len(bin_nr)):
            idcs[-(i+1)] = int(bin_nr[-(i+1)])
        idcs = tuple(idcs)
        i2p[idcs] = pos
        p2i[pos] = idcs
    return i2p, p2i


def create_interactions(dim,up_to=None):
    if up_to is None:
        up_to = dim
    interactions = {}
    i2p, _ = get_indices_position_maps(dim)
    for idxs in i2p.keys():
        if 0 < sum(idxs) <= up_to:
            interactions[idxs] = (rand()-0.5)/(sum(idxs)**2)
    return interactions

def create_random_probabilities(dim,up_to=None):
    return get_loglinear_probs(create_interactions(dim,up_to),dim)


def get_loglinear_probs(vals,dim,etas_given=False):
    i2p, p2i = get_indices_position_maps(dim)
    if etas_given:
        probs = np.zeros(2**dim)
        currOrder = dim
        while currOrder > 0:
            currTerms = filter(lambda x: sum(x) == currOrder,vals.keys())
            for idxs in currTerms:
                zeroIdxs =  filter(lambda i: idxs[i]==0,range(dim))
                probMassTerms = subsets(set(zeroIdxs))
                assignedMass = 0.0
                for flipIdxs in probMassTerms:
                    if len(flipIdxs)== 0:
                        continue
                    assignedMass = assignedMass + probs[i2p[flip(idxs,flipIdxs)]]
                probs[i2p[idxs]] = vals[idxs] - assignedMass
            currOrder = currOrder-1
        probs[0] = 1 - sum(probs)
        return probs
    else:
        expargs = np.zeros(2**dim)
        for idxs in vals.keys():
            assert(len(idxs)==dim)
            basePos = i2p[idxs]
            assert(basePos>0)
            zeroIdxs = filter(lambda i: idxs[i]==0,range(dim))
            toIncrease = subsets(set(zeroIdxs))
            for incSet in toIncrease:
                offset = sum([2**(dim-i-1) for i in incSet])
                expargs[basePos+offset] = expargs[basePos+offset] + vals[idxs]
        probs = np.exp(expargs)
        return probs/sum(probs)

def fit_log_linear_interactions(probs,up_to=None):
    dim = int(round(np.log2((probs.shape[0]))))
    if up_to is None:
        up_to = dim
    _, p2i = get_indices_position_maps(dim)
    interactions = {}
    for pos in xrange(2**dim):
        idxs = p2i[pos]
        if sum(idxs) > up_to:
            continue
        curr = np.log(probs[pos])
        oneIdxs = filter(lambda i: idxs[i]==1,range(dim))
        subPatterns = subsets(set(oneIdxs))
        for subPattern in subPatterns:
            if len(subPattern) > 0:
                offset = sum([2**(dim-i-1) for i in subPattern])
                if len(subPattern) % 2 == 0:
                    curr = curr + np.log(probs[pos-offset])
                else:
                    curr = curr - np.log(probs[pos-offset])
        interactions[idxs] = curr
    del interactions[tuple([0 for i in range(dim)])]
    return interactions

def flip(tup,idxs):
    temp = list(tup)
    return tuple([tup[i] if i not in idxs else 1-tup[i] for i in range(len(tup))])

def fit_log_linear_etas(probs,up_to=None):
    dim = int(round(np.log2((probs.shape[0]))))
    if up_to is None:
        up_to = dim
    i2p, p2i = get_indices_position_maps(dim)
    etas = {}
    for pos in xrange(1,2**dim):
        idxs = p2i[pos]
        if sum(idxs) > up_to:
            continue
        zeroIdxs = filter(lambda i: idxs[i]==0,range(dim))
        containingPatterns = subsets(set(zeroIdxs))
        etas[idxs] = sum([probs[i2p[currTup]] for currTup in map(lambda flipIdxs: flip(idxs,flipIdxs),containingPatterns)])
    return etas
        
    
    

def get_cut_model_probs(etas,interactions,k,use_scipy=True,method='hybr',seed=0):
    dim = len(interactions.keys()[0])
    etasKeys = filter(lambda keys: sum(keys) <= k, etas.keys())
    intsKeys = filter(lambda keys: sum(keys) > k,interactions.keys())
    def getPaddedBinaries(ints,minL):
        padStr = reduce(lambda x,y:x+y,['0' for i in range(minL)])
        binaries = [bin(i)[2:] for i in ints]
        return map(lambda binStr: padStr[:minL-len(binStr)]+binStr,binaries)

    varStrings = ['p'+binStr for binStr in getPaddedBinaries(xrange(2**dim),dim)]
    varSymbols = [S(varStr) for varStr in varStrings]
    eqs = []

    # all probabilities should sum to 1
    eqs.append('('+reduce(lambda x,y: x+'+'+y,varStrings)+'-1)')

    # for each marginal, add a constraint for the sum of all patterns containing the marginal indices
    for etaKey in etasKeys:
        zeroIdxs = filter(lambda i: etaKey[i]==0,range(dim))
        containingPatternsFlips = subsets(set(zeroIdxs))
        containingPatterns = map(lambda idxs: 'p'+reduce(lambda x,y: str(x)+str(y),idxs),map(lambda flipIdxs: flip(etaKey,flipIdxs),containingPatternsFlips))
        eqs.append('('+reduce(lambda x,y: x+'+'+y,containingPatterns)+'-{0})'.format(etas[etaKey]))

    # for each interaction, add a constraint
    for intKey in intsKeys:
        oneIdxs = filter(lambda i: intKey[i]==1,range(dim))
        subPatternFlips = subsets(set(oneIdxs))
        subPatterns = [flip(intKey,flipIdxs) for flipIdxs in subPatternFlips]
        expr = '(1.0'
        for i,pattern in enumerate(subPatterns):
            patternId = 'p'+reduce(lambda x,y: str(x)+str(y),pattern)
            if len(subPatternFlips[i]) % 2 ==0:
                expr = expr+'*{0}'.format(patternId)
            else:
                expr = expr+'/{0}'.format(patternId)
        eqs.append(expr+'-{0})'.format(np.exp(interactions[intKey])))
    exprs = tuple([sympify(eq) for eq in eqs])
    
    if use_scipy:
        from scipy.optimize import root,fmin_tnc,minimize,fsolve
        evaluators = [lambdify(tuple(varSymbols),expr) for expr in exprs]
        
        def objective_fun(x0):
            return np.array([evaluator(*x0) for evaluator in evaluators])
        np.random.seed(seed)
        x0 = np.random.rand(2**dim)/2**(dim-1)
        x0 = x0/np.sum(x0)
        res,idict,c,d=fsolve(objective_fun, x0,full_output=True)#,method=method)
        return res,idict,c,d
            
     
    else:
        res = nsolve(exprs,tuple(varSymbols),tuple([1.0/2**dim for _ in xrange(2**dim)]))
        return np.array(([float(r) for r in res]))
        
def get_cut_model_probs_sympy(etas,interactions,k):
    dim = len(interactions.keys()[0])
    etasKeys = filter(lambda keys: sum(keys) <= k, etas.keys())
    intsKeys = filter(lambda keys: sum(keys) > k,interactions.keys())
    def getPaddedBinaries(ints,minL):
        padStr = reduce(lambda x,y:x+y,['0' for i in range(minL)])
        binaries = [bin(i)[2:] for i in ints]
        return map(lambda binStr: padStr[:minL-len(binStr)]+binStr,binaries)

    varStrings = ['p'+binStr for binStr in getPaddedBinaries(xrange(2**dim),dim)]
    syms=symbols(reduce(lambda x,y: x+' '+y,varStrings))
    eqs = []

    # all probabilities should sum to 1
    eqs.append(reduce(lambda x,y:x+y,syms)-1)

    # for each marginal, add a constraint for the sum of all patterns containing the marginal indices
    for etaKey in etasKeys:
        zeroIdxs = filter(lambda i: etaKey[i]==0,range(dim))
        containingPatternsFlips = subsets(set(zeroIdxs))
        containingPatterns = map(lambda idxs: 'p'+reduce(lambda x,y: str(x)+str(y),idxs),map(lambda flipIdxs: flip(etaKey,flipIdxs),containingPatternsFlips))
        eqs.append(S(reduce(lambda x,y: x+'+'+y,containingPatterns))-etas[etaKey])

    # for each interaction, add a constraint
    for intKey in intsKeys:
        oneIdxs = filter(lambda i: intKey[i]==1,range(dim))
        subPatternFlips = subsets(set(oneIdxs))
        subPatterns = [flip(intKey,flipIdxs) for flipIdxs in subPatternFlips]
        expr = '1.0'
        for i,pattern in enumerate(subPatterns):
            patternId = 'p'+reduce(lambda x,y: str(x)+str(y),pattern)
            if len(subPatternFlips[i]) % 2 ==0:
                expr = expr+'*{0}'.format(patternId)
            else:
                expr = expr+'/{0}'.format(patternId)
        eqs.append(S(expr)-np.exp(interactions[intKey]))
    return eqs

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subsets(s):
    return map(set, powerset(s))

def kld(p,q):
    from IPython import embed
    mask = np.logical_and(p>0,q>0)
    return np.sum(p[mask]*np.log(p[mask]/q[mask]))

def entropy(p):
    return -sum(np.log2(p)*p)