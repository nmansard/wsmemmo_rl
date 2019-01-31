'''
In brief: run the script. You will see some reports of the algorithm activity 
on the output, and every 200 episode, a roll-out of the current policy value
is displayed (very bad in the begining, get improving). 
Change the PRE_TRAIN value to 'pre' or 'full' for loading some pre-optimized values
of the neural-network parameter. 

Train a Q-value following a classical Q-learning algorithm (enforcing the
satisfaction of HJB method), using a noisy greedy exploration strategy.

Deep Q learning, i.e. learning the Q function Q(x,u) so that Pi(x) = u = argmax Q(x,u)
is the optimal policy. The control u is discretized as 0..NU-1

This program instantiates an environment env and a Q network qvalue.
The main signals are qvalue.x (state input), qvalue.qvalues (value for any u in 0..NU-1),
qvalue.policy (i.e. argmax(qvalue.qvalues)) and qvalue.qvalue (i.e. max(qvalue.qvalue)).

The result of a training for a continuous Cozmo are stored in netvalue/qlearn_cozmo1.ckpt.

Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
Nature 518.7540 (2015): 529.
'''

from cozmomodel import Cozmo1 as Env; Env.args = { 'discretize_u': True }
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import numpy as np
import tflearn

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
tf .set_random_seed (RANDOM_SEED)
random.seed         (RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 3000          # Max training steps
NSTEPS                  = 60            # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 32            # Hidden layer size
PRE_TRAIN               = None          # None, 'pre', 'full'
ALGO_NAME               = 'qlearn'

### --- Environment
env                 = Env(**Env.args)
NX                  = env.nx            # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

class QValueNetwork:
    '''
    We represent the Q function Q(x,u) --where u is an integer [0,NU-1]-- by
    N(x)=[ Q(x,u0) ... Q(x,u_NU-1) ].
    The policy is then Pi(x) = argmax N(x) and the value function is V(x) = max Q(x)

    The classical update rule with Q:
    max_theta Q(x,u;theta) - ( reward(x,u) + decay * max_u2 Q(x2,u2) )
    with x2 = f(x,u)
    
    then rewrite as:
    max_theta N(x;theta)[u] - (reward(x,u) + decay * max N(x2)
    '''
    
    def __init__(self,NX,NU,nhiden1=32,nhiden2=32,randomSeed=None):
        if randomSeed is None:
            import time
            randomSeed = int((time.time()%10)*1000)
        n_init              = tflearn.initializations.truncated_normal(seed=randomSeed)
        u_init              = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
                                                              seed=randomSeed)
        nvars           = len(tf.trainable_variables())

        x       = tflearn.input_data(shape=[None, NX])
        netx1   = tflearn.fully_connected(x,     nhiden1, weights_init=n_init, activation='relu')
        netx2   = tflearn.fully_connected(netx1, nhiden2, weights_init=n_init)
        qvalues = tflearn.fully_connected(netx2, NU,      weights_init=u_init) # function of x only
        value   = tf.reduce_max(qvalues,axis=1)
        policy  = tf.argmax(qvalues,axis=1)

        u       = tflearn.input_data(shape=[None, 1], dtype=tf.int32)
        bsize   = tf.shape(u)[0]
        idxs    = tf.reshape(tf.range(bsize),[bsize,1])
        ui      = tf.concat([idxs,u],1)
        qvalue  = tf.gather_nd(qvalues,indices=ui)
        self.idxs = idxs
        self.ui = ui
        
        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network state   <x> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.value      = value                            # Optimal value function <Q*>
        self.policy     = policy                           # Greedy policy argmax<Q>
        self.qvalues    = qvalues                          # Q(x,.) = [ Q(x,0) ... Q(x,NU-1) ]
        self.variables  = tf.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2 ]                     # Hidden layers for debug

    def setupOptim(self,learningRate):
        qref            = tf.placeholder(tf.float32, [None])
        loss            = tflearn.mean_square(qref, self.qvalue)
        optim           = tf.train.AdamOptimizer(learningRate).minimize(loss)

        self.qref       = qref          # Reference Q-values
        self.optim      = optim         # Optimizer
        return self

    def setupTargetAssign(self,nominalNet,updateRate):
        self.update_variables = \
            [ target.assign( updateRate*ref + (1-updateRate)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self


### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

### --- Replay memory
class ReplayItem:
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2

replayDeque = deque()

### --- Tensor flow initialization
qvalue          = QValueNetwork(NX=NX,NU=NU,nhiden1=NH1,nhiden2=NH2,randomSeed=RANDOM_SEED)
qvalueTarget    = QValueNetwork(NX=NX,NU=NU,nhiden1=NH1,nhiden2=NH2,randomSeed=RANDOM_SEED)
qvalue      . setupOptim       (learningRate=QVALUE_LEARNING_RATE)
qvalueTarget. setupTargetAssign(qvalue,updateRate=UPDATE_RATE)

sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

def noisygreedy(x,rand=None):
    q = sess.run(qvalue.qvalues,feed_dict={ qvalue.x: x })
    if rand is not None: q += np.random.randn(1,env.nu)*rand
    return np.argmax(q)

def rendertrial(maxiter=NSTEPS,verbose=True):
    x = env.reset()
    rsum = 0.
    for i in range(maxiter):
        u = sess.run(qvalue.policy,feed_dict={ qvalue.x: x })
        x, reward = env.step(u)
        env.render()
        time.sleep(1e-2)
        rsum += reward
    if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

### History of search
h_rwd = []
h_ste = []    

### ---------------------------------------------------------------------------------------
### --- Training --------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

if PRE_TRAIN:
    tf.train.Saver().restore(sess, "netvalues/%s.%s.%s.ckpt" % (ALGO_NAME,str(Env), PRE_TRAIN) )
#tf.train.Saver().save(sess, "netvalues/%s.%s.%s.ckpt" % (ALGO_NAME,str(Env), PRE_TRAIN) )

for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0

    for step in range(NSTEPS):
        u       = noisygreedy(x,                                    # Greedy policy ...
                              rand=1. / (1. + episode + step))      # ... with noise
        x2,r    = env.step(u)
        done    = r>0

        replayDeque.append(ReplayItem(x,u,r,done,x2))                # Feed replay memory ...
        if len(replayDeque)>REPLAY_SIZE: replayDeque.popleft()       # ... with FIFO forgetting.

        assert( x2.shape[0] == 1 )
        
        rsum   += r
        x       = x2
        if done: break
        
        # Start optimizing networks when memory size > batch size.
        if len(replayDeque) > BATCH_SIZE:     
            batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
            x_batch    = np.vstack([ b.x      for b in batch ])
            u_batch    = np.vstack([ b.u      for b in batch ])
            r_batch    = np. stack([ b.reward for b in batch ])
            d_batch    = np. stack([ b.done   for b in batch ])
            x2_batch   = np.vstack([ b.x2     for b in batch ])

            # Compute Q(x,u) from target network
            v_batch    = sess.run(qvalueTarget.value, feed_dict={ qvalueTarget.x : x2_batch })
            qref_batch = r_batch + (d_batch==False)*(DECAY_RATE*v_batch)

            # Update qvalue to solve HJB constraint: q = r + q'
            sess.run(qvalue.optim, feed_dict={ qvalue.x    : x_batch,
                                               qvalue.u    : u_batch,
                                               qvalue.qref : qref_batch })

            # Update target networks by homotopy.
            sess.run(qvalueTarget.update_variables)

    # \\\END_FOR step in range(NSTEPS)

    # Display and logging (not mandatory).
    print('Ep#{:3d}: lasted {:d} steps, reward={:3.0f}' .format(episode, step,rsum))
    h_rwd.append(rsum)
    h_ste.append(step)
    if not (episode+1) % 200:     rendertrial(30)

# \\\END_FOR episode in range(NEPISODES)

print("Average reward during trials: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()

# Uncomment to save networks
#tf.train.Saver().save   (sess, "netvalues/qlearn_cozmo1.ckpt")
