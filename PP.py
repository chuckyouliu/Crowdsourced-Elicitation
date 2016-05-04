import numpy as np

class agent_pp:

    def __init__(self, prior_type, sig_dist):

        self.prior_type  = prior_type 
        self.sig_dist = sig_dist
        self.ntype = len(prior_type)
        self.nsignal = len(sig_dist[0])

        prior_signal = np.zeros(self.nsignal)
        for i in range(self.nsignal):
            prior_signal[i] = np.sum(self.prior_type * self.sig_dist[:,i])
        
        self.prior_signal = prior_signal

        type_given_signal = np.zeros((self.nsignal, self.ntype))
        for i in range(self.nsignal):
            for j in range(self.ntype):
                type_given_signal[i,j] = self.sig_dist[j][i]*self.prior_type[j]/self.prior_signal[i]
        self.type_given_signal = type_given_signal

    # def evaluate_prior_signal(self):

    #     prior_signal = np.zeros(self.ntype)
    #     for i in range(self.ntype):
    #         prior_signal[i] = np.sum(self.prior_type * self.sig_dist[:,i])
        
    #     self.prior_signal = prior_signal

    # def evaluate_type_given_signal(self):

    #     p1 = self.sig_dist[0][0]*self.prior_type[0]/self.prior_signal[0]
    #     p2 = self.sig_dist[0][1]*self.prior_type[0]/self.prior_signal[1]
    #     self.type_given_signal = np.array([[p1,1-p1],[p2,1-p2]])

    def signal_given_signal(self, observed, other):
        p = 0
        for i in range(self.ntype):
            p += self.type_given_signal[observed][i]*self.sig_dist[i][other]
        return p

    def update_prior_signal(self, observation):
        for i in range(self.nsignal):
            self.prior_signal[i] = self.signal_given_signal(observation, i)
        

    def prob_given_report(self, ref, self_report, ref_report):

        p = 0

        for i in range(ref.ntype):

            p += self.sig_dist[i,self_report] * (ref.sig_dist[i, ref_report] * ref.prior[i] / ref.post[ref_report])

        return p

    def report(self, u1, truth = True):

        # u1 = np.random.uniform()

        for i in range(self.ntype):
            if u1<np.sum(self.prior[:i+1]):
                signal = i
                break

        if truth:
            return i
        else:
            l = list(range(self.ntype))
            l.remove(i)
            return np.random.choice(l)

def binary_quadratic(ag_rep, ref_rep):
    if ref_rep == 0:
        return 2*ag_rep - ag_rep**2
    else:
        return 1 - ag_rep**2

# class scoring_rule:

def quadratic(a1,a1_rep,a2, a2_rep):

    r = 2*a1.prob_given_report(a2, a1_rep, a2_rep)

    for i in range(a1.ntype):

        r -= a1.prob_given_report(a2, i, a2_rep)**2

    return r

def spherical(a1,a1_rep,a2, a2_rep):

    num = a1.prob_given_report(a2, a1_rep, a2_rep)

    denom = 0

    for i in range(a1.ntype):

        denom += a1.prob_given_report(a2, i, a2_rep)**2

    return num/np.sqrt(denom)

def logarithmic(a1,a1_rep,a2, a2_rep):

    return np.log(a1.prob_given_report(a2, a1_rep, a2_rep))


def infer_report(prev, new):
    if prev>new:
        return 1
    else:
        return 0

def reward_bpp(prev, new, implicit_ref):
    return binary_quadratic(prev, implicit_ref) + binary_quadratic(new, implicit_ref)

# def set_score(self, type_):

#     if type_ == 'quadratic':
#         self.eval = quadratic
#     elif type_ == 'spherical':
#         self.eval = spherical
#     elif type_ == 'logarithmic':
#         self.eval == logarithmic
