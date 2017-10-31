import time
import json
import numpy as np
import tensorflow as tf
import augmented_lake as lake

def value_iteration(transition_model, reward_model, gamma=0.95, threshold=1e-16, max_iters=float("inf")):
    def loop_body(v, q, diff, iter_count):
        v_prev = v
        q = tf.reduce_sum(transition_model * (reward_model + (gamma * v_prev)), axis=2)
        v = tf.reduce_max(q, axis=1)
        diff = tf.reduce_max(tf.abs(v_prev - v))
        return v, q, diff, iter_count + 1

    def condition(v, q, diff, iter_count):
        return tf.logical_and(diff > threshold, iter_count < max_iters)

    nS, nA, _ = transition_model.shape
    # Initial values
    v0 = np.zeros(nS, dtype=np.float32)
    q0 = np.zeros((nS, nA), dtype=np.float32)  # Unused, but we have to pass a tensor in to get one out
    diff0 = tf.constant(float('inf'))
    iter_count0 = tf.constant(0.0)

    v, q, _, _ = tf.while_loop(condition, loop_body, [v0, q0, diff0, iter_count0])
    return v, q

def compute_policy(q_tensor, use_softmax=True):
    if use_softmax:  # Softmax is differentiable!
        return tf.nn.softmax(q_tensor)
    else:
        maxima = tf.reduce_max(q_tensor, axis=1)
        broadcast_maxima = tf.tile(tf.expand_dims(maxima, axis=1), tf.to_int32([1, tf.shape(q_tensor)[1]]))
        pi = tf.to_float(tf.equal(q_tensor, broadcast_maxima))
        return pi

def sample_trajectories(pi, starting_state, transitions, trajectory_length, num_trajectories=1000):
    def loop_body(trajectories, current_states):
        # Sample actions
        action_probabilities = tf.gather(pi, current_states)
        log_act_probs = tf.log(action_probabilities)
        sampled_actions = tf.cast(tf.multinomial(log_act_probs, num_samples=1), tf.int32)

        # Append step to trajectories
        expand_states = tf.expand_dims(current_states, axis=1)
        this_step = tf.concat([expand_states, sampled_actions], axis=1)
        expanded_step = tf.expand_dims(this_step, axis=1)
        trajectories = tf.concat([trajectories, expanded_step], axis=1)

        # Sample next states
        next_state_probs = tf.gather_nd(transitions, this_step)
        log_state_probs = tf.log(next_state_probs)
        sampled_states = tf.cast(tf.multinomial(log_state_probs, num_samples=1), tf.int32)
        next_states = tf.reshape(sampled_states, [-1])

        return trajectories, next_states

    def condition(trajectories, current_states):
        return tf.less(tf.shape(trajectories)[1], trajectory_length)

    # Initial values
    traj0 = np.zeros((num_trajectories, 0, 2), dtype=np.int32)
    states0 = tf.tile([starting_state], [num_trajectories])

    trajectories, _ = tf.while_loop(
        condition, loop_body, [traj0, states0],
        shape_invariants=[tf.TensorShape([num_trajectories, None, 2]), states0.get_shape()])
    return trajectories

def demonstrate_trajectories(env, transition_model, real_transitions, rewards, trajectory_length=50):
    v, q = value_iteration(transition_model, rewards)
    pi = compute_policy(q, use_softmax=False)

    print("With this transition model their value function on Frozen Lake looks like:")
    values = v.eval()
    print(env.arrange_on_grid(values))

    sampling = sample_trajectories(
        pi, starting_state=0, transitions=real_transitions, trajectory_length=trajectory_length)

    trajectories = sampling.eval()
    print(trajectories.shape[0], "trajectories sampled.")
    print("Here are the first three:")
    for t in range(trajectory_length):
        print('t =', t,
              '\t|   [0]:{:>2}->{:5}'.format(trajectories[0][t][0], lake.ACTION_STRINGS[trajectories[0][t][1]]),
              '  |   [1]:{:>2}->{:5}'.format(trajectories[1][t][0], lake.ACTION_STRINGS[trajectories[1][t][1]]),
              '  |   [2]:{:>2}->{:5}'.format(trajectories[2][t][0], lake.ACTION_STRINGS[trajectories[2][t][1]]))

    return trajectories

def avg_choice_log_likelihood(trajectories, policy):
    choices = np.reshape(trajectories, [-1, 2])
    gathered_likelihoods = tf.gather_nd(policy, choices)
    return tf.reduce_mean(tf.log(gathered_likelihoods))

def avg_transition_log_likelihood(trajectories, transition_model):
    outcomes = np.expand_dims(trajectories[:, 1:, 0], axis=-1)
    state_action_states = np.concatenate((trajectories[:, :-1, :], outcomes), axis=-1)
    transitions = np.reshape(state_action_states, [-1, 3])
    gathered_likelihoods = tf.gather_nd(transition_model, transitions)
    return tf.reduce_mean(tf.log(gathered_likelihoods))

def build_transition_logits_init(base_transition_model):
    base_transition_model += 1e-9  # Add epislon to the transition matrix to prevent divide by zero
    tiling = base_transition_model.shape[-1]
    rescaling = np.tile(np.expand_dims(np.sum(base_transition_model, axis=-1), axis=-1), tiling)
    base_transition_logits = np.log(base_transition_model / rescaling)
    return tf.constant_initializer(base_transition_logits)

def compute_kl_divergence(p1, p2):
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    return tf.reduce_sum(p1 * tf.log(p1 / p2), axis=-1)

def infer_transition_model(trajectories, rewards, base_transition_model=None, inference_epochs=10000, value_iter_horizon=20):
    print("Inferring the transition model of the demonstrator...")
    initializer = None if base_transition_model is None else build_transition_logits_init(base_transition_model)
    # We use logits that get softmax'd so that the variables that're optimized are in the domain (-inf,inf)
    transition_logits = tf.get_variable('inferred_transition_model', shape=rewards.shape, dtype=tf.float32, initializer=initializer)
    # The actual transition_model is a probability distribution
    transition_model = tf.nn.softmax(transition_logits)
    # Set up value iteration and use a softmax policy so there's a gradient
    v, q = value_iteration(transition_model, rewards, max_iters=value_iter_horizon)
    pi = compute_policy(q, use_softmax=True)
    # Try and maximize the likelihood of the demonstrator's choices
    choice_log_likelihood = avg_choice_log_likelihood(trajectories, pi)
    # Also try and maximize the demonstrator's correctness
    transition_log_likelihood = avg_transition_log_likelihood(trajectories, transition_model)
    # Oh and if there's a base transition model, try to minimize the distance to that
    kl_divergence = tf.constant(0)
    if base_transition_model is not None:
        kl_divergence = tf.reduce_mean(compute_kl_divergence(base_transition_model, transition_model))
    # And lastly, use L2 regularization to prevent overfitting
    L2 = tf.reduce_mean(tf.square(transition_logits))
    # Loss is a weighted sum
    transition_ll_weight = 2
    kl_weight = 0.5
    L2_weight = 0.01
    loss = -choice_log_likelihood - (transition_ll_weight * transition_log_likelihood) + (kl_weight * kl_divergence) + (L2_weight * L2)
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # Set up Tensorboard logging values
    experiment_name = str(time.time()) + '_baseline'
    tf.summary.scalar('Choice Log Likelihood', choice_log_likelihood)
    tf.summary.scalar('Transition Log Likelihood', transition_log_likelihood)
    tf.summary.scalar('KL Divergence', kl_divergence)
    tf.summary.scalar('L2', L2)
    tf.summary.scalar('loss', loss)
    merged_summaries = tf.summary.merge_all()

    # Do the gradient descent
    with tf.Session() as sess:
        logger = tf.summary.FileWriter('./data/logs/' + experiment_name, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(1, inference_epochs + 1):
            inferred_transition_model, cll, tll, kl, l2, _, summary = sess.run([
                transition_model, choice_log_likelihood, transition_log_likelihood, kl_divergence, L2, train_step, merged_summaries])
            logger.add_summary(summary, i)
            if i % 10 == 0:
                print("Epoch: {:>3}".format(i), "| Average choice LL:", cll, "| Average transition LL:", tll,
                      "| KL divergence:", kl, "| L2:", l2)

        return inferred_transition_model, v.eval()

def score(trajectories, reward_lookup):
    """ Gets the total reward for a set of trajectories """
    result = 0
    for traj in trajectories:
        for t in range(len(traj) - 1):
            s, a = traj[t]
            s_, _ = traj[t + 1]
            result += reward_lookup[s, a, s_]
    return result

def main():
    with tf.Session() as sess:
        np.set_printoptions(linewidth=120, precision=3, suppress=True)

        env = lake.AugmentedFrozenLake()
        print("Here we are on the Frozen Lake...")
        env.print_map('\t')

        rewards = env.make_rewards_tensor()
        real_transitions = env.make_real_transitions()

        # Save transition matrix
        matrix_as_list = real_transitions.tolist()
        with open('data/true_transition_matrix.js', 'w') as outfile:
            outfile.write('const trueTransitionMatrix = ' + json.dumps(matrix_as_list))

        print("The demonstrator has some unknown, naive transition model.")
        naive_transitions = env.make_naive_transitions()

        trajectories = demonstrate_trajectories(env, naive_transitions, real_transitions, rewards)

        # Save trajectories
        trajectories_as_list = trajectories.tolist()
        with open('data/demonstration_trajectories.js', 'w') as outfile:
            outfile.write('const demonstrationTrajectories = ' + json.dumps(trajectories_as_list))

        print("---------")

        inferred_transition_model, _ = infer_transition_model(trajectories, rewards, real_transitions)

        print("Here's my best guess at the transition model the agent has:")
        for s in range(inferred_transition_model.shape[0]):
            print("In state", s)
            for a in range(inferred_transition_model.shape[1]):
                print('{:>5} -> {}'.format(lake.ACTION_STRINGS[a], inferred_transition_model[s, a]))
            print()

        # Save transition model
        model_as_list = inferred_transition_model.tolist()
        with open('data/transition_model.js', 'w') as outfile:
            outfile.write('const transitionModel = ' + json.dumps(model_as_list))

        print("---------")

        new_trajectories = demonstrate_trajectories(env, inferred_transition_model, real_transitions, rewards)

        # Save new trajectories
        new_trajectories_as_list = new_trajectories.tolist()
        with open('data/inferred_model_trajectories.js', 'w') as outfile:
            outfile.write('const inferredModelTrajectories = ' + json.dumps(new_trajectories_as_list))

if __name__ == "__main__":
    main()
