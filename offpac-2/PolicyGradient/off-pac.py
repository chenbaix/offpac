import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import tflearn
import collections
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
Debug = False
Print_actor = False
print_critic = False

if "../" not in sys.path:
  sys.path.append("../")
#from lib1.envs.cliff_walking import CliffWalkingEnv
from lib1.envs.gridworld import GridworldEnv
from lib1 import plotting


X_max, Y_max = 15,15
env = GridworldEnv(shape=[X_max,X_max])
#env = CliffWalkingEnv()


class Actor():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator",lambda_trace=1.0,gamma=1.0,alpha=0.001):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.int32, shape=[],name="state")
            self.x = tf.one_hot(self.state % X_max,X_max)
            self.y = tf.one_hot(self.state / X_max,X_max)
            self.y_wbias = tf.stack(list(np.append(tf.unstack(self.y), [1])))
            self.x_wbias = tf.stack(list(np.append(tf.unstack(self.x), [1])))
            self.new_state = tf.stack([self.x_wbias,self.y_wbias])


            self.action = tf.placeholder(dtype=tf.int32 ,shape = [],name="action")
            #self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.though = tf.placeholder(dtype=tf.float32, shape=[],name="though")
            self.step = tf.placeholder(dtype=tf.int32,shape = [],name='step')


            # This is just table lookup estimator
            #self.state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))    #48 dimension with only a 1
            #self.state_wbias = tf.stack(list(np.append(tf.unstack(self.state_one_hot),[1])))   #append 1 as bias
            self.output_layer = tflearn.fully_connected(incoming=tf.expand_dims(self.new_state, 0),
                                                        #weights_init=tflearn.initializations.zeros([env.observation_space.n+1,env.action_space.n],dtype=tf.float32),
                                                        n_units = env.action_space.n,bias=False)
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))

            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            #self.picked_action_prob = tf.Variable(initial_value=0.0,dtype=tf.float32,name='picked_prob',trainable=False)
            #self.picked_prob_update = self.picked_action_prob.assign(tf.gather(self.action_probs, self.action))
            self.net_params =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)#tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=scope)[0] #get all weights
            # Loss and train op
            self.log_gradient = tf.gradients(ys=tf.log(self.picked_action_prob),xs=self.net_params)
            #self.log_gradient = tf.Variable(initial_value=tf.zeros_like(self.net_params),dtype=tf.float32,name='log_gradient',trainable=False)
            #self.log_gradient_update = self.log_gradient.assign(tf.gradients(ys=tf.log(self.picked_action_prob),xs=self.net_params))
            self.eu_pred = tf.placeholder(dtype=tf.bool,name='ev_pred')
            self.eu = tf.Variable(initial_value=tf.zeros_like(self.log_gradient), dtype=tf.float32, name='eu',
                                  trainable=False)
            #self.eu_before = tf.Variable(initial_value=tf.zeros_like(self.log_gradient),dtype=tf.float32,name='eu',trainable=False)
            #self.eu_before_update = tf.cond(self.eu_pred,
                                            #lambda: tf.assign(self.eu_before,tf.zeros_like(self.log_gradient),name='eu_before_true'),
                                            #lambda: tf.assign(self.eu_before,self.eu,name='eu_before_false'))
            self.eu_update = tf.cond(self.eu_pred,
                              lambda : tf.assign(self.eu,tf.multiply(self.though,self.log_gradient),name = 'eu_true'),
                              lambda : tf.assign(self.eu,tf.multiply(self.though, tf.add(self.log_gradient,
                                                                tf.multiply(gamma * lambda_trace ,self.eu))),name='eu_false'))
            #self.eu_update = tf.assign(self.eu,tf.multiply(self.though, tf.add(self.log_gradient,
                                                                #tf.multiply(gamma * lambda_trace ,self.eu_before))))
            self.delta = tf.placeholder(dtype=tf.float32,shape=[],name='delta')
            self.policy_gradient = tf.Variable(dtype=tf.float32,initial_value=tf.zeros_like(self.log_gradient),name='policy_gradient',trainable=False)
            self.policy_gradient_update = tf.assign(self.policy_gradient,-tf.multiply(self.eu,self.delta))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
            self.train_op = self.optimizer.apply_gradients(zip(tf.unstack(self.policy_gradient), self.net_params))


    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        if Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, action,delta, though,step,sess=None):
        sess = sess or tf.get_default_session()
        if Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        feed_dict = {self.state: state, self.delta: delta, self.though :though,self.action:action,
                     self.eu_pred:step == 1}

        if Print_actor:
            print 'step',step
            print 'action',action
            print 'state',state
            print 'though',though
            print 'delta',delta

        sess.run(self.eu_update,feed_dict)
        sess.run(self.policy_gradient_update,feed_dict)
        sess.run(self.train_op, feed_dict)
        #print 'after_gradient',sess.run(self.net_params)
        #return loss

V = []
Diff = []
V_max = []
diff_first = []
diff_second = []
class Critic():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.001, scope="value_estimator",gamma = 1.0,lambda_trace = 1.0):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32,[], "state")
            self.next_state = tf.placeholder(tf.int32, [], "next_state")
            self.this_x = tf.one_hot(self.state % X_max, X_max )
            self.this_y = tf.one_hot(self.state / X_max, X_max )
            self.thisy_wbias = tf.stack(list(np.append(tf.unstack(self.this_y), [1])))
            self.thisx_wbias = tf.stack(list(np.append(tf.unstack(self.this_x), [1])))
            self.new_state = tf.stack([self.thisx_wbias, self.thisy_wbias])
            self.next_x = tf.one_hot(self.next_state % X_max,X_max )
            self.next_y = tf.one_hot(self.next_state / X_max,X_max )
            self.nexty_wbias = tf.stack(list(np.append(tf.unstack(self.next_y), [1])))
            self.nextx_wbias = tf.stack(list(np.append(tf.unstack(self.next_x), [1])))
            self.new_next_state = tf.stack([self.nextx_wbias,self.nexty_wbias])

            #self.state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            #self.next_one_hot = tf.one_hot(self.next_state, int(env.observation_space.n))
            #self.state_wbias = tf.stack(list(np.append(tf.unstack(self.state_one_hot), [1])))
            #self.next_state_wbias = tf.stack(list(np.append(tf.unstack(self.next_one_hot), [1])))
            #self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.reward = tf.placeholder(tf.float32,[], "reward")
            self.v = tf.Variable(initial_value=tf.truncated_normal(shape=self.new_state.shape),dtype=tf.float32,name='v')
            self.v_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) #####!to be sure
            self.w = tf.Variable(initial_value=tf.truncated_normal(shape=self.new_state.shape), dtype=tf.float32, name='w')
            self.w_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)[len(self.v_param):]   ##### to be sure
            self.though = tf.placeholder(dtype=tf.float32, shape=[],name="though")
            self.step = tf.placeholder(dtype=tf.int32,name='step')
            #self.gamma = gamma
            #self.lambda_trace = lambda_trace
            self.v_cal = tf.reduce_sum(tf.reduce_sum(tf.multiply(self.v,self.new_state)))

            # This is just table lookup estimator
            #self.loss = tf.squared_difference(self.value_estimate, self.target)
            #self.diff = tf.Variable(dtype=tf.float32,expected_shape=[],name='diff',trainable=False)
            self.diff = self.reward + gamma * tf.reduce_sum(tf.reduce_sum(tf.multiply(self.v,self.new_next_state))) - \
                          tf.reduce_sum(tf.reduce_sum(tf.multiply(self.v, self.new_state)))

            self.ev_pred = tf.placeholder(dtype=tf.bool,name='ev_pred')
            self.ev_before = tf.Variable(initial_value=tf.zeros_like(self.v), dtype=tf.float32, name='ev',
                                  trainable=False)
            self.ev = tf.Variable(initial_value=tf.zeros_like(self.v), dtype=tf.float32, name='ev',
                                  trainable=False)
            self.ev_before_update = tf.cond(self.ev_pred,
                                            lambda: tf.assign(self.ev_before, tf.zeros_like(self.v),name='ev-true'),
                                            lambda: tf.assign(self.ev_before, self.ev),name='ev-false')
            self.ev_update =  tf.assign(self.ev,self.though * (tf.add(self.new_state, tf.multiply(gamma *lambda_trace , self.ev_before))))

            #self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       #decay_steps=100, decay_rate=.96, staircase=True)
            self.w_gradient = tf.Variable(initial_value=tf.zeros_like(self.w),dtype=tf.float32,name='w_gradient')
            self.v_gradient = tf.Variable(initial_value=tf.zeros_like(self.v),dtype=tf.float32,name='v_gradient')
            self.v_gradient_update = tf.assign(self.v_gradient,-tf.multiply(self.diff ,self.ev) - gamma *(1.0 - lambda_trace) * \
                                                    tf.reduce_sum(tf.reduce_sum(tf.multiply(self.w,self.ev)))* self.new_next_state)
            self.w_gradient_update = tf.assign(self.w_gradient,-tf.multiply(self.diff ,self.ev) - \
                                                            tf.reduce_sum(tf.reduce_sum(tf.multiply(self.w,self.new_state)))* self.new_state)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.w_update = self.optimizer.apply_gradients(zip([self.w_gradient], self.w_param))
            self.v_update = self.optimizer.apply_gradients(zip([self.v_gradient], self.v_param))
            #self.w_update = self.w.assign(tf.add(self.w,self.learning_rate * self.w_gradient))
            #self.v_update = self.v.assign(tf.add(self.v, self.learning_rate * self.v_gradient))
    def cal_v(self,state,sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.v_cal,{self.state:state})

    def cal_diff(self,state,next_state,reward,sess=None):
        sess = sess or tf.get_default_session()
        if Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        return sess.run(self.diff, {self.state: state,self.next_state:next_state,self.reward:reward})

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        if Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        return sess.run(tf.multiply(self.state,tf.transpose(self.v)), {self.state: state})

    def update(self, state, next_state, reward,though,step,sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.next_state: next_state,self.though:though,self.reward:reward,
                     self.ev_pred:step == 1}
        if Debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if print_critic:
            print "-"*40
            print "critic"
            print "reward",reward
            print 'step', step
            print 'next_state', next_state
            print 'state', state
            print 'though', though
            #print 'action_probs', sess.run(self.action_probs, feed_dict)
            print 'diff',sess.run(self.diff,feed_dict)

        #sess.run(self.diff_update,feed_dict)
        #v_np = tf.unstack(self.v)
        #V_max.append(np.max(sess.run(v_np,feed_dict)))
        #diff_first.append(sess.run(tf.reduce_sum(tf.multiply(self.v,self.next_state_wbias)),feed_dict))
        #diff_second.append(sess.run(tf.reduce_sum(tf.multiply(self.v,self.next_state_wbias)),feed_dict))
        #Though.append(sess.run(self.though, feed_dict))
        #Diff.append(sess.run(self.diff,feed_dict))
        #print 'diff', sess.run(self.diff, feed_dict)
        (sess.run(self.ev_before_update,feed_dict))
        (sess.run(self.ev_update,feed_dict))
        sess.run([self.w_gradient_update,self.v_gradient_update],feed_dict)
        #print "learning_rate",sess.run(self.learning_rate,feed_dict)
        #print "w_gradient",sess.run(self.w_gradient,feed_dict)
        #print "v_gradient",sess.run(self.v_gradient,feed_dict)
        #sess.run([self.w_gradient_update,self.v_gradient_update],feed_dict)
        sess.run([self.v_update],feed_dict)
        sess.run([self.w_update], feed_dict)
        #v_gradient, w_gradient = sess.run([self.v_gradient_op, self.w_gradient_op], feed_dict)
        #sess.run(tf.assign(self.v, tf.add(self.v, self.learning_rate * v_gradient)))
        #sess.run(tf.assign(self.w, tf.add(self.w , self.learning_rate * w_gradient)))

Though = []
Rewards = []
def actor_critic(env, actor, critic, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done","though"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []
        step = 0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            step += 1
            action_probs = np.squeeze(Actor.predict(state))
            #action_probs = tf.squeeze(action_probs)
            random_action = np.random.choice(np.arange(len(action_probs)))
            #action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            though = action_probs[random_action] * len(action_probs)
            next_state, _, done, _ = env.step(random_action)
            if next_state == state:
                reward = 0
            if done:
                reward = 1
            else:
                reward = -1

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=random_action, reward=reward, next_state=next_state, done=done,though=though))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            delta = critic.cal_diff(state,next_state,reward)
            # Update the value estimator
            actor.update(state, random_action,delta, though,step)

            # Update the policy estimator
            # using the td error as our advantage estimate
            critic.update(state, next_state, reward,though,step)


            # Print out which step we're on, useful for debugging.
            #print "\rStep {} @ Episode {}/{} ({})".format(
                #t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1])

            if done:
                break

            state = next_state


        if i_episode % 5 == 0:
            # One step in the environment
            test_state = env.reset()
            Reward = 0
            for t in itertools.count():
                #if t > 300:
                    #break
                # Take a step
                action_probs = np.squeeze(Actor.predict(test_state))
                # action_probs = tf.squeeze(action_probs)
                action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                #though = action_probs[random_action] * len(action_probs)
                next_test_state, _, done, _ = env.step(action)
                #env.render()
                if next_test_state == test_state:
                    reward = 0
                if done:
                    reward = 1
                else:
                    reward = -1

                # Keep track of the transition
                #episode.append(Transition(
                    #state=state, action=random_action, reward=reward, next_state=next_state, done=done, though=though))

                # Update statistics
                Reward += reward
                print "\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes,Reward)

                if done:
                    Rewards.append(Reward)
                    break
                test_state = next_test_state
        if i_episode == num_episodes - 1:
            for i in range(X_max * X_max):
                V.append(critic.cal_v(i))
            print np.array(V).reshape([X_max,X_max])

    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
Actor = Actor()
Critic = Critic()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, Actor, Critic, 1000)

#plotting.plot_episode_stats(stats, smoothing_window=10)
plt.figure()
plt.plot(Rewards)
plt.show()

#plt.figure()
#plt.plot(Diff)
#plt.show()

#plt.figure()
#plt.plot(Though)
#plt.show()

#plt.figure()
#plt.plot(diff_first)
#plt.show()

#plt.figure()
#plt.plot(diff_second)
#plt.show()