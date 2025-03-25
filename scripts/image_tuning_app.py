import os
import logging
from torchvision.io import read_image
from dotenv import load_dotenv
from torchvision.transforms.functional import resize
from modelbasedprior.prior import ModelBasedPrior
from modelbasedprior.objectives import ImageSimilarityLoss, HumanEvaluatorObjective
from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.objectives.human_evaluator.renderers import WebImageHumanEvaluatorRenderer

load_dotenv()

ORIGINAL_IMAGE_PATH = os.getenv("ORIGINAL_IMAGE_PATH")
OPTIMAL_CONFIGURATION = (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue
SEED = 23489

original_image = read_image(ORIGINAL_IMAGE_PATH)
downsampled_original_image = resize(original_image, 64)

prior_predict_func = ImageSimilarityLoss(original_image=downsampled_original_image, optimizer=OPTIMAL_CONFIGURATION, weight_psnr=0.5, weight_ssim=0.5, negate=True)

user_prior = ModelBasedPrior(bounds=prior_predict_func.bounds, predict_func=prior_predict_func, minimize=False, seed=SEED)

logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

image_renderer = WebImageHumanEvaluatorRenderer(original_image, optimal_transformation=OPTIMAL_CONFIGURATION)
human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=prior_predict_func.dim, bounds=prior_predict_func._bounds)

result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=5, logger=logger)