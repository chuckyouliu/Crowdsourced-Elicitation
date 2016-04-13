import numpy as np

class agent_pp:

    def __init__(self, prior, sig_dist):

        self.prior  = prior 
        self.sig_dist = sig_dist
        self.ntype = len(prior)
        self.post = None

    def evaluate_post(self):

        post = np.zeros(self.ntype)
        for i in range(self.ntype):
            post[i] = np.sum(self.prior * self.sig_dist[:,i])
        
        self.post = post

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

# def set_score(self, type_):

#     if type_ == 'quadratic':
#         self.eval = quadratic
#     elif type_ == 'spherical':
#         self.eval = spherical
#     elif type_ == 'logarithmic':
#         self.eval == logarithmic
