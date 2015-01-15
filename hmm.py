import numpy as np
import time

class HMM:
    def __init__(self, N, M, data, obs_indices, max_iters=1000):
        """ N is number of hidden states. M is number of observations per state."""
        self.N = N
        self.M = M
        self.data = data
        self.obs_indices = obs_indices

        # initializing the matrices suggests that all elements of each matrix be close (but not equal) to uniform

        # pi is matrix of initial probabalities 
        self.PI = np.zeros((1,N))
        self.PI.fill(1.0/N)
        noise = np.random.rand(1,N) / N
        self.PI += noise
        
        # a is matrix of transitions between states
        self.A = np.zeros((N,N))
        self.A.fill(1.0/(N*N))
        noise = np.random.rand(N,N) / (N*N)
        self.A += noise

        # b is matrix of observation probabilities given a state
        self.B = np.zeros((N,M))
        self.B.fill(1.0/(N*M))
        noise = np.random.rand(N,M) / (N*M)
        self.B += noise
        
        self.max_iters = max_iters
        self.iters = 0
        self.old_log_prob = float("-inf")
        self.log_prob = 0

    def get_index_of_observation(self,observation):
        return self.obs_indices.index(observation)

    def alpha_pass(self, observations):
        T = max(observations.shape)
        c = np.zeros((1,T))
        c[0,0] = 0
        alpha = np.zeros((T, self.N))

        for i in range(self.N):
            alpha[0,i] = self.PI[0,i] * self.B[i, self.get_index_of_observation(observations[0,0])]
            c[0,0] += alpha[0,i]
       
        c[0,0] = 1.0 / c[0,0]
        for i in range(self.N):
            alpha[0,i] = c[0,0] * alpha[0,i]

        for t in range(1,T):
            c[0,t] = 0
            for i in range(self.N):
                alpha[t,i] = 0
                for j in range(self.N):
                    alpha[t,i] += alpha[t-1, j] * self.A[j,i]
                alpha[t,i] = alpha[t,i] * self.B[i, self.get_index_of_observation(observations[0,t])]
                c[0,t] += alpha[t,i]
            
            c[0,t] = 1.0 / c[0,t]
            for i in range(self.N):
                alpha[t,i] = c[0,t] * alpha[t,i]

        return c, alpha

    def beta_pass(self, c, alpha, observations):
        T = max(observations.shape)

        beta = np.zeros((T, self.N))
        for i in range(self.N):
            beta[T-1, i] = c[0,T-1]

        for t in range(T-2, 0, -1):
            for i in range(self.N):
                beta[t,i] = 0
                for j in range(self.N):
                    beta[t,i] = beta[t,i] + self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0,t])] * beta[t+1, j]

                beta[t,i] = c[0,t] * beta[t,i]

        return beta

    def compute_gamma(self, c, alpha, beta, observations):
        T = max(observations.shape)
        gamma = np.zeros((T, self.N))
        digamma = np.zeros((T, self.N, self.N))
        for t in range(T-1):
            denom = 0
            for i in range(self.N):
                for j in range(self.N):
                    denom = denom + alpha[t,i] * self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0,t+1])] * beta[t+1,j]

            for i in range(self.N):
                gamma[t,i] = 0
                for j in range(self.N):
                    digamma[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j, self.get_index_of_observation(observations[0, t+1])] * beta[t+1,j] / denom
                    gamma[t,i] += digamma[t,i,j]

        return gamma, digamma

    def reestimate(self, gamma, digamma, observations):
        T = max(observations.shape)

        for i in range(self.N):
            self.PI[0,i] = gamma[0,i]

        # update A
        for i in range(self.N):
            for j in range(self.N):
                # have as float to guarantee float division
                numer = 0.0
                denom = 0.0
                for t in range(T-1):
                    numer += digamma[t,i,j]
                    denom += gamma[t,i]
                self.A[i,j] = numer/denom

        # update B
        for i in range(self.N):
            for j in range(self.M):
                # have as float to guarantee float division
                numer = 0.0
                denom = 0.0
                for t in range(T-1):
                    if(self.get_index_of_observation(observations[0,t]) == j):
                        numer += gamma[t,i]
                    denom += gamma[t,i]
                self.B[i,j] = numer / denom

    def compute_log(self, c, observations):
        T = max(observations.shape)
        log_prob = 0

        for t in range(T):
            log_prob += np.log(c[0,t])
        log_prob = -log_prob

        return log_prob

    def print_B(self):
        for m in range(self.M):
            for n in range(self.N):
                print self.B[n,m], 
            print ""
        
        
    def train(self):
        start_time = time.time()

        observations = self.data
        # run once in order to initialize log probability
        c, alpha = self.alpha_pass(observations)
        beta = self.beta_pass(c, alpha, observations)
        gamma, digamma = self.compute_gamma(c, alpha, beta, observations)
        self.reestimate(gamma, digamma, observations)
        self.log_prob = self.compute_log(c, observations)
    
        self.iters += 1

        # buffer allows a little bit of de-optimization in order to get out of small valleys
        buffer = 20
        while(self.iters < self.max_iters and self.log_prob > self.old_log_prob - buffer):
            then = time.time()
            self.old_log_prob = self.log_prob

            c, alpha = self.alpha_pass(observations)
            beta = self.beta_pass(c, alpha, observations)
            gamma, digamma = self.compute_gamma(c, alpha, beta, observations)
            self.reestimate(gamma, digamma, observations)
            self.log_prob = self.compute_log(c, observations)

            print "loopdeloop:", time.time()-then
            print "score:", self.log_prob
            self.iters += 1
           
        print "total run time:", time.time() - start_time
        print "number of iterations:", self.iters
