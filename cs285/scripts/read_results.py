import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        # print(e)
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob
    #tf.compat.v1.enable_eager_execution(
    #config=None, device_policy=None, execution_mode=None
    #)
    base = 'D:/Berkeley/RL/CS-285-Homework/hw3/data/'
    logdir = 'hw3_q1_MsPacman-v0_19-10-2020_09-58-10/events*'
    eventfile = glob.glob(base + logdir)[0]

    X, Y = get_section_results(eventfile)
    plt.plot(X, Y)
    plt.title('hw3_q1_MsPacman-v0: BestReturn')
    plt.show()