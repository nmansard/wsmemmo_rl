'''
Q learning with a Q-table encoded in a neural network, and a vanilla-flavor gradient
descent. There is no structure in the network, so don't expect much from this
implementation. It is given for pedagogic purpose to make the link between Q-table and
the Q-learn algorithm by Mnih and Volodymyr. 

The implementation works well for the pendulum, but fail to generalize for the car-like system.

The Q value is stored in a neural network that exactly matches the Q-Table representation (See
qtable.py):
  Q(x) = A_Q 1_x,
where A_Q is the matrix corresponding to the internal layer of the network, 
and 1_x is a vector [ 0 0  ... 0 1 0 ... 0] = onehot(x) where only the coefficient corresponding to 
the state. 

Here the representation using a neural network is overkill as it is equivalent to a table. The idea
is that the same algorithm is stil valid for any other representation, for example to handle
an environment where the state is the pixels of an image, and the first layers of the network 
are some convolutional functions.

The vector A_q 1_x is indeed the column indexed by x in A_q, i.e. A_q[x,:] in Python style.
Then the Q values for any u is given by Q(x) = A_q 1_x = [ Q(x,0) ... Q(x,NU-1) ].
The Q-value for a particular x-u pair is given by (A_q 1_x)[u] when [u] stand for the coefficient
indexed by u (integer) in the vector (A_q 1_x).

The optimal policy is the argmax of this vector: 
   Pi(x) = argmax (A_q 1_x)

As we have a network, we are in the classical neural framework where a given objective
can be optimized using the network gradient. Here we optimize the HJB residuals evaluated
at the current simulator step.
'''

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import signal
import time

from pendulummodel import DiscretePendulum as Env; Env.args = {}
# Beware: not working for the car-like robot
#from cozmomodel import Cozmo1 as Env; Env.args = { 'discretize_x': True, 'discretize_u': True }


### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 500           # Number of training episodes
NSTEPS                  = 50            # Max episode length
LEARNING_RATE           = 0.1           # Step length in optimizer
DECAY_RATE              = 0.99          # Discount factor 
PRE_TRAIN               = None          # None, "pre", "full"
ALGO_NAME               = 'deeptable'

### --- Environment
env = Env(**Env.args)
NX  = env.nx
NU  = env.nu

### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

### --- Q-value anetwork

class QValueNetwork:
    def __init__(self):
        x               = tf.placeholder(shape=[1,NX],dtype=tf.float32)
        W               = tf.Variable(tf.random_uniform([NX,NU],0,0.01,seed=100))
        qvalue          = tf.matmul(x,W)
        u               = tf.argmax(qvalue,1)

        qref            = tf.placeholder(shape=[1,NU],dtype=tf.float32)
        loss            = tf.reduce_sum(tf.square(qref - qvalue))
        optim           = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

        self.x          = x             # Network input
        self.qvalue     = qvalue        # Q-value as a function of x
        self.u          = u             # Policy  as a function of x
        self.qref       = qref          # Reference Q-value at next step (to be set to l+Q o f)
        self.optim      = optim         # Optimizer      

### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

### --- Tensor flow initialization
tf.reset_default_graph()
qvalue  = QValueNetwork()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def onehot(ix,n=NX):
    '''Return a vector which is 0 everywhere except index <i> set to 1.'''
    return np.array([[ (i==ix) for i in range(n) ],],np.float)
   
def disturb(u,i):
    u += int(np.random.randn()*10/(i/50+10))
    return np.clip(u,0,NU-1)

def rendertrial(maxiter=100):
    x = env.reset()
    for i in range(maxiter):
        u = sess.run(qvalue.u,feed_dict={ qvalue.x:onehot(x) })
        x,r = env.step(u)
        env.render()
        if r==1: print('Reward!'); break
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

### --- History of search
h_rwd = []                              # Learning history (for plot).

### ---------------------------------------------------------------------------------------
### --- Training --------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------

if PRE_TRAIN:
    tf.train.Saver().restore(sess, "netvalues/%s.%s.%s.ckpt" % (ALGO_NAME,str(Env), PRE_TRAIN) )
#tf.train.Saver().save(sess, "netvalues/%s.%s.%s.ckpt" % (ALGO_NAME,str(Env), PRE_TRAIN) )

'''
The training rationale is as follows:
- We run a number of episodes in the outward for loop. Each episod corresponds to a complete
  trial on the system, from a random state to either a success, a failure or too many steps.
- For each episode, we move the system and change the Q table accordingly.
   -1- Make a step, using the current optimized policy to compute the control u.
   -2- Change the Q-table accordingly, by trying to make the column of the Q table corresponding
       to the previous step matching the column of the next state, following the HJB equation.
'''

for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0

    for step in range(NSTEPS-1):
        u = sess.run(qvalue.u,feed_dict={ qvalue.x: onehot(x) })[0] # Greedy policy ...
        u = disturb(u,episode)                                      # ... with noise
        x2,reward = env.step(u)

        # Compute reference Q-value at state x respecting HJB
        Q2        = sess.run(qvalue.qvalue,feed_dict={ qvalue.x: onehot(x2) })
        Qref      = sess.run(qvalue.qvalue,feed_dict={ qvalue.x: onehot(x ) })
        Qref[0,u] = reward + DECAY_RATE*np.max(Q2)

        # Update Q-table to better fit HJB
        sess.run(qvalue.optim,feed_dict={ qvalue.x    : onehot(x),
                                          qvalue.qref : Qref       })

        rsum += reward
        x = x2
        if reward == 1: break

    h_rwd.append(rsum)
    if not episode%20: print('Episode #%d done with %d sucess' % (episode,sum(h_rwd[-20:])))

print("Total rate of success: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()

