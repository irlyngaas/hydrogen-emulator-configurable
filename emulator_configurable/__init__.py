from . import model_builder
from . import models
from . import train
from . import utils

# inference, data_loader, and main are intentionally NOT imported here.
# inference/data_loader depend on torchdata.datapipes (an API deprecated/
# removed in current torchdata releases), and training no longer needs
# either -- main.py imports inference lazily, inside predict_subsurface,
# only when inference is actually run. main itself imports xarray at
# module level for its own CLI-only code paths; nothing in this package
# needs it pre-imported (setup.py's console_script entry point imports
# emulator_configurable.main directly at invocation time regardless), so
# excluding it here means importing this package doesn't force xarray to
# be installed just to use, say, pi3nn's synthetic validation.

__version__  = '0.0.1'
