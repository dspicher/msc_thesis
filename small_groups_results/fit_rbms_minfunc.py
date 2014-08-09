
from scipy.optimize import fmin_cg, fmin_bfgs,fmin_ncg
import numpy as np
from amari_higher_order import *
import cPickle
from IPython import embed
from embarrassing_parallelization import *
from pyentropy.maxent import AmariSolve


def create_binary_patterns(n):
    nrs = np.arange(2**n)
    return np.bool8(np.floor(pow(2.0, np.arange(-n+1, 1, 1))*np.reshape(nrs, (-1, 1))) % 2)
    
def get_probabilities(vishid, visbiases, hidbiases, all_vis):
    # hidden units are factorized out
    logps = np.sum(all_vis*np.reshape(visbiases,(1,-1)),1)+np.sum(np.log(1+np.exp(hidbiases+np.dot(all_vis,vishid))),1)
    ps =np.exp(logps)
    return ps/np.sum(ps)
    
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
    


def objective_function(params,probs,vis,all_vis,hid,all_hid):
    visbiases = params[:vis]
    hidbiases = params[vis:vis+hid]
    vishid = np.reshape(params[vis+hid:], (vis,hid))
    return -np.sum(probs*np.log(get_probabilities(vishid,visbiases,hidbiases,all_vis)))

def gradient(params,probs,vis,all_vis,hid,all_hid):
    visbiases = params[:vis]
    hidbiases = params[vis:vis+hid]
    vishid = np.reshape(params[vis+hid:], (vis,hid))
    
    factor = np.reshape(probs, (-1, 1))

    # positive phase
    hidden_probs = factor*sigmoid(np.dot(all_vis, vishid)+np.reshape(hidbiases, (1, -1)))
    gradient_w_pos = np.dot(all_vis.T, hidden_probs)
    gradient_h_pos = np.sum(hidden_probs, 0)
    gradient_v_pos = np.sum(factor*all_vis, 0)

    # negative phase, we factorize over the hidden units
    visinput = np.dot(all_hid,vishid.T) + np.reshape(visbiases, (1, -1))
    visactivation = sigmoid(visinput)
    prob_hidden_patterns = np.prod(np.exp(all_hid*np.reshape(hidbiases, (1,-1))), 1)*np.prod(1+np.exp(visinput),1)
    prob_hidden_patterns = np.reshape(prob_hidden_patterns/np.sum(prob_hidden_patterns), (-1, 1))
    gradient_w_neg = np.dot((prob_hidden_patterns*visactivation).T, all_hid)
    gradient_h_neg = np.sum(prob_hidden_patterns*all_hid, 0)
    gradient_v_neg = np.sum(prob_hidden_patterns*visactivation,0)

    vishidgrad = gradient_w_pos - gradient_w_neg
    visbiasesgrad = gradient_v_pos - gradient_v_neg
    hidbiasesgrad = gradient_h_pos - gradient_h_neg

    return -np.concatenate((visbiasesgrad,hidbiasesgrad,vishidgrad.flatten()))
    

def run_random_interactions((rep_i,(probs,hid,order,name,idx)),num_starts=5):
    vis = 6
    all_vis = create_binary_patterns(vis)
    all_hid = create_binary_patterns(hid)
    am_sol = AmariSolve(vis,2)
    res = {}
    res['orig_probs'] = probs
    res['hid'] = hid
    res['order'] = order
    from IPython import embed
    #embed()
    me_probs = am_sol.solve(probs[1],order)
    res['me_probs'] = me_probs
    res['return'] = np.zeros(num_starts)
    res['xopt'] = np.zeros((num_starts,vis+hid+vis*hid))
    res['final_obj_func'] = np.zeros(num_starts)
    res['final_gradient'] = np.zeros((num_starts,vis+hid+vis*hid))
    res['function_calls'] = np.zeros((num_starts,2))
    for seed in range(num_starts):
        print seed
        np.random.seed(seed)
        vishid = np.random.randn(vis, hid)/np.sqrt(vis+hid)
        visbiases = np.random.randn(vis)
        hidbiases = np.random.randn(hid)
        x0 = np.concatenate((visbiases,hidbiases,vishid.flatten()))
        #embed()
        xopt,fopt, func_calls,grad_calls,warnflag = fmin_cg(objective_function,x0,maxiter=20000,fprime=gradient,norm=np.Inf,gtol=1e-7,args=(me_probs,vis,all_vis,hid,all_hid),full_output=True,disp=True)
        res['return'][seed] = warnflag #0: success, 1: maxiter, 2: no convergence
        res['xopt'][seed,:] = xopt
        res['final_obj_func'][seed] = objective_function(xopt,me_probs,vis,all_vis,hid,all_hid)
        res['final_gradient'][seed,:] = gradient(xopt,me_probs,vis,all_vis,hid,all_hid)
        res['function_calls'][seed,:] = [func_calls, grad_calls]
    print "{0} succesful for ({2},{3},{4})".format(np.sum(res['return']==0), vis, hid, order)
    cPickle.dump(res,open('{0}_{1}_order_{2}_hid_{3}.p'.format(name,idx,order,hid),'wb'))
    
if __name__=='__main__':
    import cPickle
    groups = cPickle.load(open('groups_of_6_culture_retina_50.p','rb'))
    params = []
    names = ['Culture','Retina']
    for nameIdx,name in enumerate(names):
        keys = list(groups[nameIdx])
        for keyIdx,key in enumerate(keys):
            for order in range(1,7):
                for hid in range(1,8):
                    params.append((groups[nameIdx][key],hid,order,name,keyIdx))
        cPickle.dump(keys,open('{0}_key_list.p'.format(name),'wb'))
    run_tasks(1,params,run_random_interactions,withmp=False)
    
