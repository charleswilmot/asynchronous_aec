class Conf:
    """Meant to contain all parameters related to the model.
    todo: add the ratios parameter, that controls which ratios are used
    todo: pass this conf object directly to the constructor of the Worker object
    todo: buffer size should not be fixed (=20) but should be defined from the command line
    """
    def __init__(self, args):
        self.mlr, self.clr = args.model_learning_rate, args.critic_learning_rate
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.reward_scaling_factor = reward_scaling_factor
        self.discount_factor = args.discount_factor
        self.episode_length = args.episode_length
        self.update_factor = args.update_factor
        self.buffer_size = 2000
        self.batch_size = args.batch_size
