from utils.summary import generate_summary, evaluate_summary
from exceptions import InvalidEvaluationMethod, InExistentLossInEvaluation, InvalidEvaluationMetric


class EvaluationMetric:
    configs = [
            {
                "method": "ground_truth_max",
                "metrics": ['f_measure_by_max', "precision_by_max", "recall_by_max"],
                "optimal": "max"
            },
            {
                "method": "ground_truth_avg",
                "metrics": ['f_measure_by_avg', "precision_by_avg", "recall_by_avg"],
                "optimal": "max"
            },
            {
                "method": "loss",
                "metrics": ["loss_*"],
                "optimal": "min"
            },
            {
                "method": "f1",
                "metrics": ["f1"],
                "optimal": "max"
            }

        ]

    @staticmethod
    def get_evaluation_configs():
        return EvaluationMetric.configs

    @staticmethod
    def get_evaluation_optimal(metric):
        for config in EvaluationMetric.configs:
            for i, _metric in enumerate(config['metrics']):
                if _metric == metric:
                    if type(config['optimal']) == str:
                        return config['optimal']
                    else:
                        return config['optimal'][i]
                break
        raise InvalidEvaluationMetric(metric)

    @staticmethod
    def calculate_metrics(scores, batch, eval_metric):
        image_features, video_names, change_points, num_frames, nfps, positions, user_summary = batch[:7]
        change_points = change_points[0].numpy()
        num_frames = num_frames[0]
        nfps = nfps
        nfps = list(map(lambda x: int(x[0]), nfps))
        positions = positions[0]
        user_summary = user_summary[0].numpy()
        positions = positions.view(-1).numpy()
        scores = scores.detach().cpu().numpy()
        machine_summary = generate_summary(scores, change_points, num_frames, nfps, positions)
        return evaluate_summary(machine_summary, user_summary, eval_metric)




    @staticmethod
    def extract_loss(losses, loss_name):
        return losses[loss_name]

    @staticmethod
    def generate_metrics(evaluation_methods, scores, losses, probs, batch):
        metrics = {

        }
        for evaluation_method in evaluation_methods:
            if evaluation_method == "ground_truth_max":
                fm, prec, rec = EvaluationMetric.calculate_metrics(scores, batch, "max")
                metrics['f_measure_by_max'] = fm
                metrics['precision_by_max'] = prec
                metrics['recall_by_max'] = rec
            elif evaluation_method == "ground_truth_avg":
                fm, prec, rec = EvaluationMetric.calculate_metrics(scores, batch, "avg")
                metrics['f_measure_by_avg'] = fm
                metrics['precision_by_avg'] = prec
                metrics['recall_by_avg'] = rec
            elif evaluation_method.startswith("loss_"):
                loss_name = evaluation_method[5:]
                if loss_name in losses:
                    metrics[loss_name] = EvaluationMetric.extract_loss(losses, loss_name)
                else:
                    raise InExistentLossInEvaluation(loss_name)
            else:
                raise InvalidEvaluationMethod(evaluation_method)

        return metrics
