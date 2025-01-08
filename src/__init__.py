import sys
import sys
sys.path.append('/mnt/suhas/OCR/trocr_train/src')
from main import TrocrPredictor

# expose the TrocrPredictor interface to other models
__all__ = ["TrocrPredictor"]
