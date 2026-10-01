from . import model_builder
from . import models
from . import train
from . import utils
from . import main

# inference and data_loader are intentionally NOT imported here -- both
# depend on torchdata.datapipes (an API deprecated/removed in current
# torchdata releases), and training no longer needs either. main.py imports
# inference lazily, inside predict_subsurface, only when inference is
# actually run.

__version__  = '0.0.1'
