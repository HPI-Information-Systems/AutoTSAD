import json
import pathlib


_MODULE_DIR = pathlib.Path(__file__).parent.absolute()


def validate_input_map_spec(obj):
    # type: List[str]
    assert "type" in obj
    assert obj["type"] in ["pixels", "dct", "conv", "random_weights", "gradient", "compose"]

    # input_scale: float, optional
    if "input_scale" in obj:
        assert isinstance(obj["input_scale"], float)

    # operations: List[obj], optional
    if "operations" in obj:
        assert isinstance(obj["operations"], list)
        for op in obj["operations"]:
            validate_input_map_spec(op)

    # xsize if pixels
    # ksize if dct
    # kernel_shape if conv
    # hidden_size if random_weights

    # size: List[int], optional
    if "size" in obj:
        assert isinstance(obj["size"], list)
        for s in obj["size"]:
            assert isinstance(s, int)
    # kernel_type: str, optional
    if "kernel_type" in obj:
        assert obj["kernel_type"] in ["mean", "gauss", "random"]
    # mode: str, optional
    if "mode" in obj:
        assert obj["mode"] in ["same", "same"]


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path=None, params=None):
        if json_path is not None and params is not None:
            raise ValueError("json_path and params are sopedually exclusive args")

        if json_path is not None:
            with open(str(json_path)) as f:
                self.__dict__ = json.load(f)

        if params is not None:
            self.__dict__ = params

    def _required_of_type(self, name, tpe):
        assert name in self.__dict__
        assert isinstance(self.__dict__[name], tpe)
    
    def validate(self):
        # input_shape: List[int]
        self._required_of_type("input_shape", list)

        # input_map_specs: List[Dict]
        self._required_of_type("input_map_specs", list)
        for spec in self.__dict__["input_map_specs"]:
            validate_input_map_spec(spec)

        # reservoir_representation: str (sparse, dense)
        self._required_of_type("reservoir_representation", str)
        assert self.__dict__["reservoir_representation"] in ["sparse", "dense"]

        # spectral_radius: float
        self._required_of_type("spectral_radius", float)
        # density: float
        self._required_of_type("density", float)

        # train_length: int
        self._required_of_type("train_length", int)
        # pred_length: int
        self._required_of_type("pred_length", int)
        # transient_length: int
        self._required_of_type("transient_length", int)

        # train_method: str
        self._required_of_type("train_method", str)
        assert self.__dict__["train_method"] in ["pinv_svd", "pinv_lstsq", "tikhonov"]

        # tikhonov_beta: Optional[float] = None
        self.__dict["tikhonov_beta"] = self.__dict__.get("tikhonov_beta", None)
        # imed_loss: str
        self._required_of_type("imed_loss", bool)

        # imed_sigma: Optional[float] = None
        self.__dict["imed_sigma"] = self.__dict__.get("imed_sigma", None)

        # backend: str
        self._required_of_type("backend", str)
        assert self.__dict__["backend"] in ["numpy"]

        # dtype: str
        self._required_of_type("dtype", str)
        assert self.__dict__["dtype"] in ["float32", "float64"]

        # debug: str
        self._required_of_type("debug", bool)


        # anomaly_start: int, optional
        if "anomaly_start" in self.__dict__:
            assert isinstance(self.__dict__["anomaly_start"], int)
        # anomaly_step: int, optional
        if "anomaly_step" in self.__dict__:
            assert isinstance(self.__dict__["anomaly_step"], int)
        # timing_depth: int, optional
        if "timing_depth" in self.__dict__:
            assert isinstance(self.__dict__["timing_depth"], int)

    def save(self, json_path):
        with open(str(json_path), 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, params):
        """Updates parameters based on a dictionary or a list."""
        if isinstance(params, list):
            for i in range(0, len(params), 2):
                key, value = params[i], params[i + 1]
                try:
                    value = eval(value)
                except Exception:
                    pass
                self.__dict__[key] = value
        else:
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        ps = self.__dict__.copy()
        maps = ps.pop("input_map_specs")

        def maps_table_row(imap):
            if 'input_scale' in imap:
                scale = f"{imap['input_scale']:.2f}"
            else:
                scale = " -- "

            if imap['type'] in ['pixels', 'dct', 'random_weights']:
                maps_row = f"  {imap['type']:<8} {scale:<3} {imap['size']}\n"
            elif imap['type'] == 'conv':
                maps_row = f"  {imap['type']:<8} {scale:<3} {str(imap['size']):<9} {imap['kernel_type']}\n"
            elif imap['type'] == 'gradient':
                maps_row = f"  {imap['type']:<8} {scale:<3}\n"
            elif imap['type'] == 'compose':
                ops = imap['operations']
                maps_row = "  compose\n"
                for m in ops:
                    maps_row += f"  {maps_table_row(m)}"
            return maps_row

        maps_table = ""
        for imap in maps:
            maps_table += maps_table_row(imap)

        ps_dump = json.dumps(ps, indent=4, sort_keys=True)
        return f"Params:\n{ps_dump}\nInput maps:\n{maps_table}"


def default_params():
    json_path = _MODULE_DIR / "default_params.json"
    return Params(json_path=json_path)
