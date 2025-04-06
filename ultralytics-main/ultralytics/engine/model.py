# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(torch.nn.Module):
    """
    一个用于实现 YOLO 模型的基类，统一了不同模型类型的 API。

    该类为与 YOLO 模型相关的各种操作提供了一个通用接口，例如训练、验证、预测、导出和基准测试。它处理不同类型的模型，包括从本地文件、Ultralytics HUB 或 Triton Server 加载的模型。

    属性:
        callbacks (Dict): 一个包含模型操作期间各种事件回调函数的字典。
        predictor (BasePredictor): 用于进行预测的预测器对象。
        model (torch.nn.Module): 底层的 PyTorch 模型。
        trainer (BaseTrainer): 用于训练模型的训练器对象。
        ckpt (Dict): 如果模型是从 *.pt 文件加载的，则包含检查点数据。
        cfg (str): 如果模型是从 *.yaml 文件加载的，则包含模型配置。
        ckpt_path (str): 检查点文件的路径。
        overrides (Dict): 模型配置的覆盖参数字典。
        metrics (Dict): 最新的训练/验证指标。
        session (HUBTrainingSession): Ultralytics HUB 会话（如果适用）。
        task (str): 模型所针对的任务类型。
        model_name (str): 模型的名称。

    方法:
        __call__: predict 方法的别名，使模型实例可以直接调用。
        _new: 根据配置文件初始化一个新模型。
        _load: 从检查点文件加载模型。
        _check_is_pytorch_model: 确保模型是 PyTorch 模型。
        reset_weights: 将模型的权重重置为初始状态。
        load: 从指定文件加载模型权重。
        save: 将模型的当前状态保存到文件。
        info: 记录或返回模型信息。
        fuse: 融合 Conv2d 和 BatchNorm2d 层以优化推理。
        predict: 执行目标检测预测。
        track: 执行目标跟踪。
        val: 在数据集上验证模型。
        benchmark: 在各种导出格式上对模型进行基准测试。
        export: 将模型导出为不同格式。
        train: 在数据集上训练模型。
        tune: 执行超参数调优。
        _apply: 对模型的张量应用函数。
        add_callback: 为事件添加回调函数。
        clear_callback: 清除事件的所有回调函数。
        reset_callbacks: 将所有回调函数重置为默认函数。
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = True,
    ) -> None:

        print("\nModel类__init__()被调用","地址：ultralytics-main\\ultralytics\\engine\\model.py")

        """
        初始化 YOLO 模型类的新实例。

        此构造函数根据提供的模型路径或名称设置模型。它处理各种类型的模型来源，包括本地文件、Ultralytics HUB 模型和 Triton Server 模型。该方法初始化模型的几个重要属性，并为其准备训练、预测或导出等操作。

        参数:
            model (Union[str, Path]): 要加载或创建的模型的路径或名称。可以是本地文件路径、Ultralytics HUB 的模型名称或 Triton Server 模型。
            task (str | None): 与 YOLO 模型关联的任务类型，指定其应用领域。
            verbose (bool): 如果为 True，则在模型初始化及其后续操作期间启用详细输出。

        异常:
            FileNotFoundError: 如果指定的模型文件不存在或无法访问。
            ValueError: 如果模型文件或配置无效或不支持。
            ImportError: 如果特定模型类型（如 HUB SDK）所需的依赖项未安装。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model = Model("path/to/model.yaml", task="detect")
            >>> model = Model("hub_model", verbose=True)
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()  # 获取默认回调函数
        self.predictor = None  # 重用预测器
        self.model = None  # 模型对象
        self.trainer = None  # 训练器对象
        self.ckpt = {}  # 如果从 *.pt 文件加载
        self.cfg = None  # 如果从 *.yaml 文件加载
        self.ckpt_path = None  # 检查点路径
        self.overrides = {}  # 训练器配置的覆盖参数
        self.metrics = None  # 验证/训练指标
        self.session = None  # HUB 会话
        self.task = task  # 任务类型
        self.model_name = None  # 模型名称
        model = str(model).strip()  # 去除前后空格

        # 检查是否是来自 https://hub.ultralytics.com 的 Ultralytics HUB 模型
        if self.is_hub_model(model):
            # 从 HUB 获取模型
            checks.check_requirements("hub-sdk>=0.0.12")
            session = HUBTrainingSession.create_session(model)
            model = session.model_file
            if session.train_args:  # 如果是从 HUB 发送的训练任务
                self.session = session

        # 检查是否是 Triton Server 模型
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # 如果未明确设置，则默认任务为检测
            return

        # 加载或创建新的 YOLO 模型
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 避免确定性警告
        if Path(model).suffix in {".yaml", ".yml"}:  # 如果是 YAML 配置文件
            self._new(model, task=task, verbose=verbose)
        else:  # 否则加载模型
            self._load(model, task=task)

        # 删除 super().training 以便访问 self.model.training
        del self.training




############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################





    @staticmethod
    def is_triton_model(model: str) -> bool:
        print('Model.is_triton_model()被调用', '检查提供的模型是否是 Triton Server 模型')
        """
        Checks if the given model string is a Triton Server URL.

        This static method determines whether the provided model string represents a valid Triton Server URL by
        parsing its components using urllib.parse.urlsplit().

        Args:
            model (str): The model string to be checked.

        Returns:
            (bool): True if the model string is a valid Triton Server URL, False otherwise.

        Examples:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            True
            >>> Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}
############################################################################################################################################################################################

    @staticmethod
    def is_hub_model(model: str) -> bool:
        print("\nModel.is_hub_model()被调用",'检查提供的模型是否是 Ultralytics HUB 模型')

        """
        检查提供的模型是否是 Ultralytics HUB 模型。

        此静态方法用于判断给定的模型字符串是否代表一个有效的 Ultralytics HUB 模型标识符。

        参数:
            model (str): 要检查的模型字符串。

        返回:
            (bool): 如果模型是有效的 Ultralytics HUB 模型，则返回 True，否则返回 False。

        示例:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/MODEL")
            True
            >>> Model.is_hub_model("yolo11n.pt")
            False
        """
        return model.startswith(f"{HUB_WEB_ROOT}/models/")
############################################################################################################################################################################################


    def _load(self, weights: str, task=None) -> None:

        print("\nModel._load()被调用",'加载的权重设置模型、任务和相关属性')

        """
        从检查点文件加载模型或从权重文件初始化模型。

        该方法处理从 .pt 检查点文件或其他权重文件格式加载模型。它根据加载的权重设置模型、任务和相关属性。

        参数:
            weights (str): 要加载的模型权重文件路径。
            task (str | None): 与模型关联的任务类型。如果为 None，则从模型中推断。

        异常:
            FileNotFoundError: 如果指定的权重文件不存在或无法访问。
            ValueError: 如果权重文件格式不受支持或无效。

        示例:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights
############################################################################################################################################################################################



    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,     # 在预测任务中，默认加载为<ultralytics.models.yolo.detect.predict.DetectionPredictor object>对象
        **kwargs: Any,
    ) -> List[Results]:

        print('\nModel.predict()被调用','进行预测,即将返回结果(List[ultralytics.engine.results.Results])')

        """
        使用 YOLO 模型对给定的图像源进行预测。

        该方法通过关键字参数支持各种配置，简化了预测过程。它支持使用自定义预测器或默认预测器方法进行预测。该方法处理不同类型的图像源，并可以在流模式下运行。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 
                要进行预测的图像来源。接受各种类型，包括文件路径、URL、PIL 图像、numpy 数组和 torch 张量。
            stream (bool): 如果为 True，则将输入源视为连续流进行预测。
            predictor (BasePredictor | None): 用于进行预测的自定义预测器类的实例。如果为 None，则使用默认预测器。
            **kwargs: 用于配置预测过程的额外关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 预测结果列表，每个结果封装在一个 Results 对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # 打印检测框数据

        注意:
            - 如果未提供 'source'，则默认使用 ASSETS 常量并发出警告。
            - 如果预测器尚未设置，则该方法会设置一个新的预测器，并在每次调用时更新其参数。
            - 对于 SAM 类型的模型，可以通过关键字参数传递 'prompts'。
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )#判断是否是命令行调用的yolo或者ultralytics，并且是否包含predict或者track，如果是，则is_cli为True

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # 配置默认的预测参数
        #置信度阈值0.25，batch大小为1，保存结果由is_cli决定，模式为predict
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        #合并三组参数，优先级从低到高为： self.overrides < custom < kwargs
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            #运行my_detect.py时，会进入这个if语句，此时self.predictor为None
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            #如果 predictor 为 None ，则调用 self._smart_load("predictor")
            #使用配置参数 args 和回调函数 _callbacks 初始化预测器 predictor
            #设置预测器模型并配置是否显示详细信息（由 is_cli 决定）
            print('\nModel.predictor类属性被初始化','设置预测器模型为：',type(self.predictor))
            self.predictor.setup_model(model=self.model, verbose=is_cli)#self.model=DetectionModel类

        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)

        #运行my_detect.py时，不会进入这个if语句    
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)


        #self.predictor 默认为<ultralytics.models.yolo.detect.predict.DetectionPredictor object at 0x000001F7268FCE50>
        
        if  is_cli:
            return self.predictor.predict_cli(source=source)
        else:  #推理时默认is_cli=False
            result=self.predictor(source=source, stream=stream)#返回一个结果列表，每个结果是一个Results对象
            #ultralytics.engine.results.Results object list
            #自定义img_data
            img_data=result[0].boxes.data    #数据为torch.Tensor，shape为[1, 6]，表示检测框的坐标、置信度、类别
            return result

       
############################################################################################################################################################################################


    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        在加载 PyTorch 模型检查点时重置特定参数。

        该静态方法过滤输入参数字典，仅保留一组被认为对模型加载重要的键。
        它用于确保在从检查点加载模型时只保留相关参数，丢弃任何不必要或可能冲突的设置。

        参数:
            args (dict): 包含各种模型参数和设置的字典。

        返回:
            (dict): 一个新的字典，仅包含输入参数中指定的键。

        示例:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

############################################################################################################################################################################################




    def _smart_load(self, key: str):
        print("\nModel._smart_load()被调用 根据模型任务加载相应的模块")
        """
        根据模型任务加载相应的模块。

        该方法根据模型当前任务和提供的 key 动态选择并返回正确的模块（model、trainer、validator 或 predictor）。
        它使用 task_map 属性来确定要加载的正确模块。

        参数:
            key (str): 要加载的模块类型。必须是 'model'、'trainer'、'validator' 或 'predictor' 之一。

        返回:
            (object): 与指定 key 和当前任务对应的已加载模块。

        异常:
            NotImplementedError: 如果当前任务不支持指定的 key。

        示例:
            >>> model = Model(task="detect")
            >>> predictor = model._smart_load("predictor")
            >>> trainer = model._smart_load("trainer")

        注意:
            - 该方法通常由 Model 类的其他方法内部使用。
            - task_map 属性应正确初始化，包含每个任务的正确映射。
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e
############################################################################################################################################################################################

    def __getattr__(self, name):
        """
        Enables accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)
            >>> print(model.task)
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################


    @property
    def task_map(self) -> dict:
        """
        提供从模型任务到不同模式下对应类的映射。

        该属性方法返回一个字典，将每个支持的任务（例如检测、分割、分类）映射到一个嵌套字典。
        嵌套字典包含不同操作模式（model、trainer、validator、predictor）到它们各自类实现的映射。

        该映射允许根据模型的任务和所需的操作模式动态加载适当的类。这为处理 Ultralytics 框架中的各种任务和模式提供了灵活且可扩展的架构。

        返回:
            (Dict[str, Dict[str, Any]]): 一个字典，其中键是任务名称（str），值是嵌套字典。
            每个嵌套字典包含 'model'、'trainer'、'validator' 和 'predictor' 键，映射到它们各自的类实现。

        示例:
            >>> model = Model()
            >>> task_map = model.task_map
            >>> detect_class_map = task_map["detect"]
            >>> segment_class_map = task_map["segment"]

        注意:
            该方法的实际实现可能因 Ultralytics 框架支持的具体任务和类而异。
            文档字符串提供了预期行为和结构的一般描述。
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """
        Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >> model = YOLO("yolo11n.pt")
            >> model.eval()
        """
        self.model.eval()
        return self





    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        print("使用了Model的call方法")#检测任务中并未使用
        
        """
        predict 方法的别名，使模型实例可以直接调用进行预测。

        该方法通过允许直接调用模型实例来简化预测过程，只需传入必要的参数即可。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): 
                要进行预测的图像来源。可以是文件路径、URL、PIL 图像、numpy 数组、PyTorch 张量或这些类型的列表/元组。
            stream (bool): 如果为 True，则将输入源视为连续流进行预测。
            **kwargs: 用于配置预测过程的额外关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 预测结果列表，每个结果封装在一个 Results 对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """

        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:#推理时未被调用

        print("Model._new()被调用")

        """
        初始化一个新模型并从模型定义中推断任务类型。

        该方法根据提供的配置文件创建一个新的模型实例。它加载模型配置，如果未指定任务类型则从配置中推断，并使用任务映射中的适当类初始化模型。

        参数:
            cfg (str): YAML 格式的模型配置文件路径。
            task (str | None): 模型的具体任务类型。如果为 None，则从配置中推断。
            model (torch.nn.Module | None): 自定义模型实例。如果提供，则使用该实例而不是创建新模型。
            verbose (bool): 如果为 True，则在加载期间显示模型信息。

        异常:
            ValueError: 如果配置文件无效或无法推断任务类型。
            ImportError: 如果指定任务所需的依赖项未安装。

        示例:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg


    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        使用指定的数据集和验证配置对模型进行验证。

        该方法简化了模型验证过程，允许通过各种设置进行自定义。它支持使用自定义验证器或默认验证方法进行验证。
        该方法结合了默认配置、方法特定的默认值和用户提供的参数来配置验证过程。

        参数:
            validator (ultralytics.engine.validator.BaseValidator | None): 用于验证模型的自定义验证器类的实例。
            **kwargs: 用于自定义验证过程的任意关键字参数。

        返回:
            (ultralytics.utils.metrics.DetMetrics): 从验证过程中获得的验证指标。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # 打印 mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics#
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to customize the benchmarking process. These are combined with
                default configurations, model-specific arguments, and method defaults. Common options include:
                - data (str): Path to the dataset for benchmarking.
                - imgsz (int | List[int]): Image size for benchmarking.
                - half (bool): Whether to use half-precision (FP16) mode.
                - int8 (bool): Whether to use int8 precision mode.
                - device (str): Device to run the benchmark on (e.g., 'cpu', 'cuda').
                - verbose (bool): Whether to print detailed benchmark information.
                - format (str): Export format name for specific benchmarking

        Returns:
            (Dict): A dictionary containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose", False),
            format=kwargs.get("format", ""),
        )



    def _check_is_pytorch_model(self) -> None:#推理时未被调用

        print("Model._check_is_pytorch_model()被调用",'检查模型是否为PyTorch模型')
        """
        检查模型是否为 PyTorch 模型，如果不是则抛出 TypeError 异常。

        该方法验证模型是否为 PyTorch 模块或 .pt 文件。用于确保需要 PyTorch 模型的特定操作仅在兼容的模型类型上执行。

        异常:
            TypeError: 如果模型不是 PyTorch 模块或 .pt 文件。错误信息会详细说明支持的模型格式和操作。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # 不会抛出异常
            >>> model = Model("yolo11n.onnx")
            >>> model._check_is_pytorch_model()  # 抛出 TypeError 异常
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )
    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs: Arbitrary keyword arguments to customize the export process. These are combined with
                the model's overrides and method defaults. Common arguments include:
                format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                half (bool): Export model in half-precision.
                int8 (bool): Export model in int8 precision.
                device (str): Device to run the export on.
                workspace (int): Maximum memory workspace size for TensorRT engines.
                nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def reset_weights(self) -> "Model":#推理时未被调用

        """
        Resets the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True,
        enabling them to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":#推理时未被调用
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (Union[str, Path]): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as
        the date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (Union[str, Path]): The name of the file to save the model to.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output and return the information as a list.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information. If False, returns the information as a list.

        Returns:
            (List[str]): A list of strings containing various types of information about the model, including
                model summary, layer details, and parameter counts. Empty if verbose is True.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # Prints model summary
            >>> info_list = model.info(detailed=True, verbose=False)  # Returns detailed info as a list
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers
        into a single layer. This fusion can significantly improve inference speed by reducing the number of
        operations and memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Raises:
            TypeError: If the model is not a PyTorch torch.nn.Module.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:#推理时未被调用

        print("Model.embed()被调用",'图像源source生成嵌入')

        """
        基于提供的图像源生成图像嵌入。

        该方法是 'predict()' 方法的封装，专注于从图像源生成嵌入。它允许通过各种关键字参数自定义嵌入过程。

        参数:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): 
                用于生成嵌入的图像源。可以是文件路径、URL、PIL 图像、numpy 数组等。
            stream (bool): 如果为 True，则进行流式预测。
            **kwargs: 用于配置嵌入过程的额外关键字参数。

        返回:
            (List[torch.Tensor]): 包含图像嵌入的列表。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)



    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        使用注册的跟踪器对指定的输入源进行目标跟踪。

        该方法使用模型的预测器和可选的注册跟踪器执行目标跟踪。它处理各种输入源，如文件路径或视频流，并支持通过关键字参数进行自定义。如果尚未存在跟踪器，该方法会注册跟踪器，并可以在不同调用之间持久化跟踪器。

        参数:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): 
                用于目标跟踪的输入源。可以是文件路径、URL 或视频流。
            stream (bool): 如果为 True，则将输入源视为连续视频流。默认为 False。
            persist (bool): 如果为 True，则在不同调用之间持久化跟踪器。默认为 False。
            **kwargs: 用于配置跟踪过程的额外关键字参数。

        返回:
            (List[ultralytics.engine.results.Results]): 跟踪结果列表，每个结果都是一个 Results 对象。

        异常:
            AttributeError: 如果预测器没有注册的跟踪器。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # 打印跟踪 ID
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)




    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.

        Raises:
            ValueError: If the event name is not recognized or is invalid.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]








    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach. The method handles scenarios such as resuming training
        from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
        arguments and warns if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process.

        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            **kwargs: Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.

        Returns:
            (Dict | None): Training metrics if available and training is successful; otherwise, None.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path


        #self.trainer 为<ultralytics.models.yolo.detect.train.DetectionTrainer object at 0x00000290BF536CD0>
        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # 如果不是从检查点恢复训练
            # 手动设置模型：如果存在检查点则使用当前模型作为权重，否则使用默认配置
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model  # 更新模型引用

        self.trainer.hub_session = self.session  # 附加可选的 HUB 会话 self.session默认为None
        self.trainer.train()  # 开始训练过程
        
        # 训练完成后更新模型和配置
        if RANK in {-1, 0}:  # 仅在主进程（rank为-1或0）执行
            # 获取最佳或最后的检查点
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            # 加载检查点权重
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            # 更新模型参数
            self.overrides = self.model.args
            # 获取验证器指标（DDP模式下可能没有返回指标）
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: DDP 模式下没有返回指标
        return self.metrics  # 返回训练指标

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args: Variable length argument list for additional arguments.
            **kwargs: Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (Dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(use_ray=True, iterations=20)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)



    def _apply(self, fn) -> "Model":
        """
        Applies a function to model tensors that are not parameters or registered buffers.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like
        moving the model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like
                to(), cpu(), cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (Dict[int, str]): A dict of class names associated with the model.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.

        Returns:
            (torch.device): The device (CPU/GPU) of the model.

        Raises:
            AttributeError: If the model is not a torch.nn.Module instance.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None
