# napari-pyfast-image-processing (npyFAST)

[![License](https://img.shields.io/pypi/l/napari-pyfast-image-processing.svg?color=green)](https://github.com/haesleinhuepf/napari-pyfast-image-processing/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-pyfast-image-processing.svg?color=green)](https://pypi.org/project/napari-pyfast-image-processing)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-pyfast-image-processing.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-pyfast-image-processing/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-pyfast-image-processing/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-pyfast-image-processing/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-pyfast-image-processing)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-pyfast-image-processing)](https://napari-hub.org/plugins/napari-pyfast-image-processing)

Use [pyFAST's](https://fast.eriksmistad.no) CPU/GPU-accelerated image processing from within napari. 
Available functions are yet limited to some simple operations such as `gaussian_blur, image_gradient, laplacian_of_gaussian_2D, image_sharpening_2D, dilation, erosion`.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


## Installation

First make sure that the requirements of FAST are [installed as described here](https://fast.eriksmistad.no/requirements.html).

<!--
On MacOS this can be done by running these commands from the command line:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
/opt/homebrew/bin/brew install openslide libomp
```
-->

Afterwards, you can install `napari-pyfast-image-processing` via [pip]:

<!--
```
pip install napari-pyfast-image-processing
```


To install latest development version :
-->

```
pip install git+https://github.com/haesleinhuepf/napari-pyfast-image-processing.git
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-pyfast-image-processing" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-pyfast-image-processing/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
