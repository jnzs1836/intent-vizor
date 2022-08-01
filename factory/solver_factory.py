from solvers import QueryFocusedSolver, QueryFocusedDeploymentSolver, ShotQueryFocusedSolver
from exceptions import InvalidSolverException, InvalidModelException
from .model_factory import build_summarizer, build_compressor, build_discriminator, build_critic
from .model_factory import build_topic_aware, build_topic_absent, build_shot_query_topic_aware, build_shot_query_baseline, build_shot_random_guess, build_shot_linear_baseline
import pickle


def open_pickle_file(filename):
    fp = open(filename, "rb")
    data = pickle.load(fp)
    return data
    with (open(filename, "rb")) as openfile:
        t = pickle.load(openfile)
        while True:
            try:
                t = pickle.load(openfile)
                print(t)
                return t
            except EOFError:
                print(EOFError)
                break


def get_difference_attention(dataset):
    return open_pickle_file(dataset)
    # with open("summe_diff_attention.pickle", "rb") as fp:
    #    return pickle.load(fp)

def build_gan_solver(config):
    solver = GANSolver(config)
    summarizer = build_summarizer(config)
    discriminator = build_discriminator(config)
    compressor = build_compressor(config)
    solver.build(compressor, summarizer, discriminator)
    return solver





def build_query_deployment_solver(config):
    score_type = "concept-wise"
    if config.solver == "QueryFocus-MonoScore":
        score_type = "mono"
    scheduler_milestones = config.scheduler_milestones
    solver = QueryFocusedDeploymentSolver(config, score_type=score_type, milestones=scheduler_milestones)
    if config.summarizer == "TopicAware":
        model = build_topic_aware(config)
    elif config.summarizer == "TopicAbsent":
        model = build_topic_absent(config)
    else:
        model = None
    solver.build(model)
    return solver


def build_query_focused_solver(config):
    score_type = "concept-wise"
    if config.solver == "QueryFocus-MonoScore":
        score_type = "mono"
    # milestones_str = config.scheduler_milestones.split(",")
    # scheduler_milestones = list(map(lambda x: int(x), milestones_str))
    scheduler_milestones = config.scheduler_milestones
    solver = QueryFocusedSolver(config, score_type=score_type, milestones=scheduler_milestones)
    if config.summarizer == "TopicAware":
        model = build_topic_aware(config)
    elif config.summarizer == "TopicAbsent":
        model = build_topic_absent(config)
    else:
        model_config = {
            "similarity_dim": 1000,
            "concept_dim": 300,
            "in_channel": 2048,
            "conv1_channel": 512,
            "conv2_channel": 256,
            "deconv1_channel": 1024,
            "deconv2_channel": 1024,
            "max_segment_num": 20,
            "max_frame_num": 200,
            "device": config.device
        }

        model = build_chan(model_config)
    solver.build(model)
    return solver


def build_shot_query_focused_solver(config):
    score_type = "mono"
    scheduler_milestones = config.scheduler_milestones
    solver = ShotQueryFocusedSolver(config, score_type=score_type, milestones=scheduler_milestones)
    if config.summarizer == "ShotQueryTopic":
        model = build_shot_query_topic_aware(config)
    elif config.summarizer == "TopicAbsent":
        model = build_topic_absent(config)
    elif config.summarizer == "linear_baseline":
        model = build_shot_linear_baseline(config)
    elif config.summarizer == "baseline":
        model = build_shot_query_baseline(config)
    elif config.summarizer == "random":
        model = build_shot_random_guess(config)

    solver.build(model)
    return solver


def build_solver(config):
    if config.solver == "GAN":
        return build_gan_solver(config)
    elif config.solver.startswith("QueryFocus"):
        return build_query_focused_solver(config)
    elif config.solver.startswith("ShotQueryFocus"):
        return build_shot_query_focused_solver(config)
    else:
        raise InvalidSolverException(config.solver)
