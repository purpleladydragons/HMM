import numpy as np

# number of hidden states
N = 3 

# number of observations (40 for english phonemes)
M = 40

class HMM:
    def __init__(self, N, M, data, max_iters=100):
        self.N = N
        self.M = M
        self.data = data

        # TODO turn from zeros into random stochastic-row matrices
        # TODO can't be uniform, but should be close to it
        # pi is matrix of initial probabalities
        self.PI = np.zeros((1,N))
        # a is matrix of transitions between states
        self.A = np.zeros((N,N))
        # b is matrix of observation probabilities given a state
        self.B = np.zeros((N,M))
        
        self.max_iters = max_iters
        self.iters = 0
        self.old_log_prob = float("-inf")

    def alpha_pass(self, observations):
        c = np.zeros((1,N))
        c[0,0] = 0
        T = len(observations)
        alpha = np.zeros((T, N))
        

        for i in range(self.N):
            alpha[0,i] = self.PI[0,i] * self.B[0,self.get_index_of_observation(observations[0,0])]
            c[0,0] += alpha[0,i]
       
        c[0,0] = 1.0 / c[0,0]
        for i in range(self.N):
            alpha[0,i] = c[0,0] * alpha[0,i]

        for t in range(1,T):
            c[0,t] = 0
            for i in range(self.N):
                alpha[t,i] = 0
                for j in range(self.N):
                    # TODO make sure A is proper for this format
                    alpha[t,i] += alpha[t-1, j] * self.A[j,i]
                # TODO this is another potentially problematic line (confusing) liek this is almsot definitely wrong or different from line 35
                alpha[t,i] = alpha[t,i] * self.B[i, self.get_index_of_observation(observations[0,t])]
                c[0,t] += alpha[t,i]
            
            c[0,t] = 1.0 / c[0,t]
            for i in range(N):
                alpha[t,i] = c[0,t] * alpha[t,i]

        return c, alpha

    def beta_pass(self, c, alpha, observations):
        T = len(observations)

        beta = np.zeros((T, self.N))
        for i in range(N):
            beta[T-1, i] = c[0,T-1]

        for t in range(T-2, 0, -1):
            for i in range(N):
                beta[t,i] = 0
                for j in range(N):
                    beta[t,i] = beta[t,i] + self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0,t])] * beta[t+1, j]

                beta[t,i] = c[0,t] * beta[t,i]

        return beta

    def compute_gamma(self, c, alpha, beta, observations):
        T = len(observations)
        gamma = np.zeros((T, self.N))
        digamma = np.zeros((T, self.N, self.N))
        for t in range(T-1):
            denom = 0
            for i in range(N):
                for j in range(N):
                    denom = denom + alpha[t,i] * self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0,t+1])] * beta[t+1,j]

            for i in range(N):
                gamma[t,i] = 0
                for j in range(N):
                    digamma[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0, t+1])] * beta[t+1,j] / denom
                    gamma[t,i] += digamma[t,i,j]

        return gamma, digamma

    def reestimate(self, gamma, observations):
        T = len(observations)

        for i in range(self.N):
            self.PI[0,i] = gamma[0,i]

        # update A
        for i in range(self.N):
            for j in range(self.N):
                numer = 0
                denom = 0
                for t in range(T-1):
                    numer += digamma[t,i,j]
                    denom += gamma[t,i]
                # TODO floatify?
                self.A[i,j] = numer/denom

        # update B
        for i in range(self.N):
            for j in range(self.M):
                numer = 0
                denom = 0
                for t in range(T-1):
                    if(self.get_index_of_observration(observations[0,t]) == j):
                        numer += gamma[t,i]
                    denom += gamma[t,i]
                self.B[i,j] = numer / denom

    def compute_log(self, c, observations):
        T = len(observations)
        log_prob = 0

        for t in range(T):
            log_prob += np.log(c[0,i])
        log_prob = -log_prob

        return log_prob
        
    # TODO this whole while loop structure doesn't work with the log probability comparison
    def train(self, observations):
        while(self.iters < self.max_iters and self.log_prob > self.old_log_prob):
            self.alpha_pass(observations)
            self.beta_pass()
            self.compute_gamma()
            self.reestimate()
            self.compute_log()

            self.old_log_prob = self.log_prob
            self.iters += 1
            
