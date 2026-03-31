import copy
import math
try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np

try:
    from Tools import phi, reverse_phi
except:
    from .Tools import phi, reverse_phi
        
class openloop_inputs():
    def __init__(self,inputs):
        self.inputs=inputs

    def sample(self,n):
        try:
            input=self.inputs[:,n]
            termination=False
        except IndexError:
            input=None
            termination=True
        return input,termination,None

    def update(self,beh=None):
        pass


class RnnModel():
    def __init__(self, dtData=0.186, dt=0.001, tau=0.1, g=1, N=500,
                 phi='modifiedtanh',input_dim=11,output_dim=None,observation_dim=None,max_length=200):
        #model settings
        self.tau = tau
        self.dtData = dtData
        self.dt = dt
        self.td_t = int(dtData/dt)
        self.g = g  # initial weight scale factor
        self.phi = phi

        #model structure
        self.N = N
        self.input_dim = input_dim #11 for Y-maze
        self.observation_dim = observation_dim #5 for Y-maze
        self.output_dim = output_dim #3 for Y-maze
        self.max_length = max_length

    def set_target(self, neuron_target, neuron_target_type='x',behavior_target=None,inputs=None,true_env_states=None,x0=None):
        """
        Args:
            neuron_target (ndarray of shape (N,T)):
            behavior_target (ndarray of shape (out_dim,T)) 
            x0: if None then reverse_phi from r0
        """
        self.neuron_target = neuron_target  # firing rate
        self.neuron_target_type = neuron_target_type
        self.behavior_target = behavior_target
        self.inputs=inputs #inputs to the network
        self.true_env_states=true_env_states #true environment states (if available)
        self.n_datapoint = self.neuron_target.shape[1]
        self.x0 = x0
 
    def initialize_params(self):
        """
        initialize weight matrix
        """
        J_in = np.zeros((self.N, self.input_dim))
        J = self.g * np.random.randn(self.N, self.N) / math.sqrt(self.N)
        np.fill_diagonal(J, 0)
        self.J_all = np.hstack((J, J_in))
        if self.output_dim is not None:
            self.J_out = np.zeros((self.output_dim, self.N))
        else:
            self.J_out = None

    def set_params(self, J_all,J_out=None):
        self.J_all = J_all
        self.J_out = J_out

    def _begin_network_state(self):
        if self.x0 is None:
            if self.neuron_target_type == 'r':
                x0 = reverse_phi(self.neuron_target[:, 0], self.phi)
            else:
                x0 = self.neuron_target[:, 0]
        else:
            x0 = self.x0
        self.R = np.zeros((self.N, self.max_length))
        if self.neuron_target_type == 'r':
            self.R[:self.N, 0] = self.neuron_target[:, 0]
        else:
            self.R[:self.N, 0] = phi(self.neuron_target[:, 0],self.phi)
        self.X = np.zeros((self.N, self.max_length))
        self.X[:self.N, 0] = x0
        self.U = np.zeros((self.input_dim, self.max_length)) #observation inputs
        self.P = np.zeros((self.observation_dim, self.max_length)) #true environment states (if available)
        self.P[:,0]=self.true_env_states[:,0]
        if self.output_dim is not None:
            self.X_out = np.zeros((self.output_dim, self.max_length))
            self.X_out[:, 0] = self.behavior_target[:, 0]
        return x0

    def _save(self, n, x, u, p, beh=None):
        # save the nth data to memory
        r = phi(x, self.phi)
        self.R[:, n] = copy.deepcopy(r)
        self.X[:, n] = copy.deepcopy(x)
        self.U[:, n] = copy.deepcopy(u)
        if p is not None:
            self.P[:, n] = copy.deepcopy(p)
        elif beh is not None:
            self.P[:self.output_dim,n] = copy.deepcopy(beh)
        if beh is not None:
            self.X_out[:, n] = copy.deepcopy(beh)
    
    def _step(self, x, input, external_input=0):
        # forward pass
        u_vec = np.hstack((phi(x, self.phi), input))
        u = np.dot(self.J_all, u_vec) 
        x = x + self.dt / self.tau * (u - x + np.random.randn(self.N) * self.cur_noise + external_input)
        if self.J_out is not None:
            beh = np.dot(self.J_out, phi(x, self.phi))
        else:
            beh = None
        return x, u_vec, beh        

    def run(self, cur_noise=0,closed_loop=False,input_generator=None):
        """
        input_generator must be customized for closed-loop env.
        In close-loop setup, it contains an instance of an environment.
        In open-loop setup, it's simply an input array (as specified in set_target),
          and the generator (automatically instantiated) would just sample from it sequentially.
        """
        self.cur_noise = cur_noise
        x0 = self._begin_network_state()
        if closed_loop is False:
            input_generator=openloop_inputs(self.inputs)
        n = 0
        for i in range(self.max_length*self.td_t):
            if i == 0:
                x = x0
            else:
                input, termination, true_env_states = input_generator.sample(n)
                if termination is not None:  
                    if termination:# hit end of maze
                        break
                x, u_vec, beh = self._step(x, input[:self.input_dim])

                #only needed when it's closed-loop simulation
                input_generator.update(beh)
            
            if i >= self.td_t*(n + 1):
                self._save(n+1, x, input, true_env_states, beh)
                n = n + 1
                
        if self.output_dim is not None:
            return self.R[:, :n + 1], self.X[:, :n + 1],self.U[:, :n + 1],self.P[:,:n+1],self.X_out[:, :n + 1]
        else:
            return self.R[:, :n + 1], self.X[:, :n + 1],self.U[:, :n + 1],self.P[:,:n+1]
    
    def run_CLV(self,K=100,norm=0.1,renorm_period=1,closed_loop=False,input_generator=None,
                    CLV=False,max_length=None):
        """
        Numerically estimate the first K Lyapunov exponents.
        input_generator must be customized for closed-loop env.
        """        
        def QR_householder(A):
            Q,R=np.linalg.qr(A)
            n, k = A.shape
            for i in range(k):
                if R[i,i]<0:
                    R[i,:]=R[i,:]*(-1)
                    Q[:,i]=Q[:,i]*(-1)
            return Q,R

        if max_length is None:
            max_length = self.max_length
        self.cur_noise = 0
        x0 = self._begin_network_state()
        if closed_loop is False:
            input_generator=openloop_inputs(self.inputs)
        n = 0
        ratios=np.empty((0,K))
        A=np.random.randn(self.N,K)
        Q,_=QR_householder(A)
        Qs=np.empty((0,self.N,K))
        Qs=np.vstack((Qs,Q[np.newaxis,:,:]))
        Rs=np.empty((0,K,K))
        renorm_count=0
        
        for i in range(max_length*self.td_t):
            if i == 0:
                x = x0
                x_errors=x0[:,np.newaxis]+Q*norm
                # copies of env is techiqally only needed for closed-loop environment, 
                # but for code readability, we generalize the implementation 
                # to both closed-loop and open-loop setup 
                input_errors={}
                for k in range(K):
                    input_errors[k]=copy.deepcopy(input_generator)
            else:
                input, termination,_ = input_generator.sample(n)
                if termination is not None:  
                    if termination:# hit end of maze
                        break
                x,_,beh = self._step(x, input[:self.input_dim])
                input_generator.update(beh)

                for k in range(K):
                    # for closed-loop env, input_k would be different from input
                    # for open-loop, they are the same
                    input_k, _,_ = input_errors[k].sample(n)
                    x_k, _, beh_k = self._step(x_errors[:,k], input_k[:self.input_dim])
                    x_errors[:,k]=copy.deepcopy(x_k)
                    input_errors[k].update(beh_k)
                
            if i >= self.td_t*(n + 1):
                n = n + 1
                renorm_count += 1
            if renorm_count >= renorm_period:
                renorm_count = 0
                A=x_errors-x[:,np.newaxis]
                Q,R=QR_householder(A)
                ratios=np.vstack((ratios,np.einsum('nk,nk->k',Q,A)[np.newaxis,:]/norm))
                Qs=np.vstack((Qs,Q[np.newaxis,:,:]))
                Rs=np.vstack((Rs,R[np.newaxis,:,:]))
                x_errors=x[:,np.newaxis]+Q*norm
                input_errors={}
                for k in range(K):
                    input_errors[k]=copy.deepcopy(input_generator)
            if n+1>=max_length:
                break

        if CLV is True:
            #backward process
            C=np.eye(K)
            Cs=np.empty((0,K,K))
            Cs=np.vstack((C[np.newaxis,:,:],Cs))
            for i in range(Rs.shape[0]):
                C=np.matmul(np.linalg.inv(Rs[-1-i]/norm),Cs[-1-i])
                C=C/np.linalg.norm(C,axis=0)
                Cs=np.vstack((C[np.newaxis,:,:],Cs))
            Vs=np.einsum('tnk,tkj->tnj',Qs,Cs)
            return ratios,Qs,Vs

        else:
            return ratios,Qs
    
    def train(self, optim_neurons, optim_outputs=None, numEpoch=1,cur_noise=0, 
              closed_loop=False,input_generator=None,target_align=None,generalizedTF=False,alpha=0):
        """
        In closed-loop env, besides the input_genertor, a target_align method also needs to be provided.
        For example, in the mouse navigation case, the method align simulated and 
        data trajectories according to the forward position of mouse.
        In open-loop env, target alignment is not needed since the simulated and data trajectories are already aligned by time.
        """
        self.cur_noise = cur_noise
        assert optim_neurons.P.shape[0] == self.J_all.shape[1], \
            'to-neuron optimizer does not match the shape of to-neuron weights'
        if optim_outputs is not None:
            assert optim_outputs.P.shape[0] == self.J_out.shape[1], \
                'outputs optimizer does not match the shape of output weights'
        if closed_loop is False:
            input_generator=openloop_inputs(self.inputs)
        
        for nEpoch in range(numEpoch):
            x0 = self._begin_network_state()
            n = 0
            for i in range(self.max_length*self.td_t):
                if i == 0:
                    x = x0
                else:
                    input, termination, true_env_states = input_generator.sample(n)
                    if termination is not None:  
                        if termination:# hit end of maze
                            break
                    x, u_vec, beh = self._step(x, input[:self.input_dim])
                    input_generator.update(beh)

                if i >= self.td_t*(n + 1):
                    if closed_loop is True:
                        t_target = target_align(self.true_env_states,self.P[:,n])
                       # t_target = target_align(self.behavior_target,self.inputs,self.true_env_states,
                       #                         input,true_env_states)
                    else:
                        t_target = n+1
                        if t_target >= self.neuron_target.shape[1]:  # if run out of training time steps
                            break
                    neu_target = self.neuron_target[:, t_target]
                    if self.neuron_target_type == 'x':
                        error=x-neu_target
                    else:
                        error=phi(x, self.phi) - neu_target 
                    self.J_all=optim_neurons.update(self.J_all,
                                                    self.dt / self.tau * u_vec,
                                                    error)
                    if self.J_out is not None:
                        beh_target = self.behavior_target[:, t_target]
                        self.J_out=optim_outputs.update(self.J_out,
                                                        phi(x, self.phi),
                                                        beh - beh_target[:self.output_dim])
                    #if generalizedTF is True:
                    #    # generalized teacher forcing
                    #    if self.neuron_target_type == 'x':
                    #        x = alpha*neu_target+(1-alpha)*x
                    #    else:
                    #        x = alpha*reverse_phi(neu_target, self.phi)+(1-alpha)*x
                    self._save(n+1, x, input, true_env_states, beh)
                    n = n + 1
        return n
