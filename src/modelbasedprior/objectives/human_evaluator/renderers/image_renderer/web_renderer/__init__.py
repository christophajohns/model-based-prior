from .web_renderer import WebImageHumanEvaluatorRenderer

if __name__ == "__main__":
    import logging
    import torch
    from modelbasedprior.logger import setup_logger
    from modelbasedprior.prior import ModelBasedPrior
    from modelbasedprior.objectives.human_evaluator.human_evaluator_objective import HumanEvaluatorObjective
    from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
    from modelbasedprior.optimization.bo import maximize

    logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    # Image-based evaluation
    original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    image_similarity = ImageSimilarityLoss(original_image=original_image, negate=True)
    user_prior = ModelBasedPrior(
        bounds=image_similarity.bounds,
        predict_func=image_similarity,
        minimize=False,
    )
    image_renderer = WebImageHumanEvaluatorRenderer(original_image)
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=image_similarity.dim, bounds=image_similarity._bounds)

    result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=5, logger=logger)