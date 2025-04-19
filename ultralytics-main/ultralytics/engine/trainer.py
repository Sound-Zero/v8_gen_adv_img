# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
import matplotlib.pyplot as plt

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)


        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.trainset, self.testset = self.get_dataset()    #self.trainset, self.testset 均为str类型
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        """允许在多GPU系统中使用device=''或device=None，默认使用device=0。"""
        # 计算world_size（使用的GPU数量）
        if isinstance(self.args.device, str) and len(self.args.device):  # 例如 device='0' 或 device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # 例如 device=[0, 1, 2, 3]（从CLI传入的列表）
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # 例如 device='cpu' 或 'mps'
            world_size = 0  # 使用CPU或MPS时world_size为0
        elif torch.cuda.is_available():  # 例如 device=None 或 device='' 或 device=number
            world_size = 1  # 默认使用device 0
        else:  # 例如 device=None 或 device=''
            world_size = 0

        # 如果是DDP训练且未设置LOCAL_RANK，则运行子进程
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # 参数检查
            if self.args.rect:  # rect模式与多GPU训练不兼容
                LOGGER.warning("警告 ⚠️ 'rect=True' 与多GPU训练不兼容，已设置为 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:  # batch小于1与多GPU训练不兼容
                LOGGER.warning(
                    "警告 ⚠️ 'batch<1' 与多GPU训练不兼容，已设置为默认 'batch=16'"
                )
                self.args.batch = 16

            # 生成并运行DDP命令
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} 调试命令 {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))  # 清理DDP资源

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        """在正确的rank进程上构建数据加载器和优化器。"""
        # 模型初始化
        self.run_callbacks("on_pretrain_routine_start")  # 执行预训练回调
        ckpt = self.setup_model()  # 设置模型
        self.model = self.model.to(self.device)  # 将模型移动到指定设备
        self.set_model_attributes()  # 设置模型属性

        # 冻结层处理
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)  # 如果是列表，直接使用
            else range(self.args.freeze)  # 如果是整数，生成范围
            if isinstance(self.args.freeze, int)  # 检查是否为整数
            else []  # 否则返回空列表
        )
        always_freeze_names = [".dfl"]  # 总是冻结这些层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names  # 生成要冻结的层名列表

        # 遍历模型参数
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # 将NaN转为0（注释掉，因为会导致训练结果不稳定）
            if any(x in k for x in freeze_layer_names):  # 如果层名在冻结列表中
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False  # 设置requires_grad为False
            elif not v.requires_grad and v.dtype.is_floating_point:  # 如果层未冻结且是浮点类型
                LOGGER.info(
                    f"警告 ⚠️ 为冻结层 '{k}' 设置 'requires_grad=True'。"
                    "参见 ultralytics.engine.trainer 了解如何自定义冻结层。"
                )
                v.requires_grad = True  # 设置requires_grad为True

        # 检查AMP（自动混合精度）
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True或False
        if self.amp and RANK in {-1, 0}:  # 单GPU和DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调，因为check_amp()会重置它们
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # 恢复回调
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # 从rank 0广播tensor到所有其他rank（返回None）
        self.amp = bool(self.amp)  # 转换为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:  # 如果是多GPU训练
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # 检查图像尺寸
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格尺寸（最大步长）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # 用于多尺度训练

        # 批量大小
        if self.batch_size < 1 and RANK == -1:  # 仅单GPU，估计最佳批量大小
            self.args.batch = self.batch_size = self.auto_batch()

        # 数据加载器
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")






        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # 优化器相关设置
        # 计算梯度累积步数,防止显存溢出
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        # 根据batch size和累积步数缩放权重衰减
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        # 计算总迭代次数
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        # 构建优化器
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum, 
            decay=weight_decay,
            iterations=iterations,
        )
        # 设置学习率调度器
        self._setup_scheduler()
        # 初始化早停机制
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        # 恢复训练
        self.resume_training(ckpt)
        # 设置调度器的last_epoch
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        # 运行预训练例程结束回调
        self.run_callbacks("on_pretrain_routine_end")
    def _do_train(self, world_size=1):
        """主要训练循环"""
        # 分布式训练设置
        if world_size > 1:
            self._setup_ddp(world_size)  # 初始化分布式训练
        self._setup_train(world_size)    # 设置训练所需组件
    
        # 训练参数初始化
        nb = len(self.train_loader)  # 总批次数量
        # 计算warmup迭代次数（前nw个batch进行学习率热身）
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  
        last_opt_step = -1  # 最后优化步记录
        # 计时相关初始化
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        # 训练开始回调
        self.run_callbacks("on_train_start")
        # 打印训练信息
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        
        # Mosaic增强关闭设置
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清空梯度
        
        # 主训练循环
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")  # 周期开始回调





            
            ###############————————自定义————————############################################################################################################

            attack_task=""#"fgsm"      #  "fgsm"  or  "pgd"
            if attack_task=="fgsm":
                all_imgs = []
                grads=[]
                img_path=[]
                epsilons = [  0.25]
                #保存不同epsilon的对抗样本到指定绝对路径
                save_adv_path=r"D:\\MyPytonProject\\pythonProject\\YOLOv8\\adv_images"
            elif attack_task=="pgd":
                all_imgs = []
                grads=[]
                img_path=[]
                iter_num=5  #前向传播次数
                alpha=0.05  #每次扰动幅度
                epsilon=0.25  #总体的扰动幅度绝对值
                save_adv_path=r"D:\\MyPytonProject\\pythonProject\\YOLOv8\\adv_images"
            ###################################################################################################################################



            
            # 学习率调度
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()  # 更新学习率
    
            # 设置模型为训练模式
            self.model.train()
            # 分布式训练设置
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            # 数据加载器设置
            pbar = enumerate(self.train_loader)
            # 关闭Mosaic增强
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
    
            # 主进程进度条设置
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            self.tloss = None  # 初始化总损失
            
            # 批次循环
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")  # 批次开始回调
                
                # Warmup阶段设置s
                ni = i + nb * epoch  # 全局迭代次数
                if ni <= nw:
                    xi = [0, nw]  # 插值范围
                    # 计算梯度累积次数
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    # 参数组学习率调整
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 偏置参数的学习率调整
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
    
                # 前向传播
                with autocast(self.amp):  # 自动混合精度上下文
                    
                    





                    batch = self.preprocess_batch(batch)  # 数据预处理



                    #############################——————自定义—————########################################################################################################################
                    # 原为计算损失
                    if attack_task=='pgd' and epoch == self.epochs-1:

                        if os.path.exists(save_adv_path)==False:
                            print(f"不存在{save_adv_path}无法保存图片")
                            return

                        orig_img=batch["img"].detach()
                        img_path.append(batch["im_file"])#list类型,每个元素为str类型,表示绝对路径
                        adv_img=orig_img.clone()
                        for iter_idx in range(iter_num):
                            min_val,max_val=torch.min(adv_img).item(),torch.max(adv_img).item()


                            
                            self.loss, self.loss_items = self.model(batch)  # 计算损失
                            my_grad=torch.autograd.grad(outputs=self.loss, inputs=batch["img"],grad_outputs=torch.ones_like(self.loss),create_graph=True)
                            
     
                            adv_img=adv_img +alpha * my_grad[0].sign()
                            # 限制扰动范围
                            delta = torch.clamp(adv_img - orig_img, min=-epsilon, max=epsilon)

                            # 进行下一轮的对抗样本生成
                            adv_img = orig_img + delta
                            adv_img = torch.clamp(adv_img, min=0, max=1).detach()
                            batch["img"]=adv_img
                            batch["img"].requires_grad=True


                            self.optimizer.zero_grad()  # 清空梯度


                        
                        if os.path.exists(save_adv_path):
                            target_path=""
                            if not os.path.exists(os.path.join(save_adv_path,f"pgd_eps{epsilon}")):
                                os.makedirs(os.path.join(save_adv_path,f"pgd_eps{epsilon}"))
                                target_path=os.path.join(save_adv_path,f"pgd_eps{epsilon}")
                            elif os.path.exists(os.path.join(save_adv_path,f"pgd_eps{epsilon}")):
                                target_path=os.path.join(save_adv_path,f"pgd_eps{epsilon}")


                            # 保存图片
                            print("保存结果图")
                            for index in range(len(adv_img)):
                                img_name= os.path.basename(img_path[0][index])
                                one_img=adv_img[index].permute(1,2,0).detach().cpu().numpy()
                                plt.imsave(os.path.join(target_path,f"{img_name}"),one_img)
                            print("保存")

                    else:
                        self.loss, self.loss_items = self.model(batch)  # 计算损失


                    if attack_task=='fgsm':
                        if epoch == self.epochs-1:
                            all_imgs.append(batch["img"].detach())#tensor类型
                            img_path.append(batch["im_file"])#list类型,每个元素为str类型,表示绝对路径
                            
                            my_grad=torch.autograd.grad(outputs=self.loss, inputs=batch["img"],grad_outputs=torch.ones_like(self.loss),create_graph=True)#,retain_graph=True
                            grads.append(my_grad)
                    ###################################################################################################################################################################




  




                    if RANK != -1:  # 分布式训练损失缩放
                        self.loss *= world_size
                    # 计算平均损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )


                self.scaler.scale(self.loss).backward()  # 反向传播<torch.cuda.amp.grad_scaler.GradScaler object at 0x000002DAE86732E0>
 


                # 参数优化
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()  # 执行优化步骤
                    last_opt_step = ni  # 更新最后优化步
    
                    # 时间控制停止
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # 分布式训练同步停止状态
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:  # 训练超时则停止
                            break
    
                # 日志记录
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    # 更新进度条显示
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length)) % (
                            f"{epoch + 1}/{self.epochs}",  # 当前epoch
                            f"{self._get_memory():.3g}G",  # GPU显存使用
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失值
                            batch["cls"].shape[0],  # 当前批次大小
                            batch["img"].shape[-1],  # 图像尺寸
                        )
                    )
                    self.run_callbacks("on_batch_end")  # 批次结束回调
                    # 绘制训练样本
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
    
                self.run_callbacks("on_train_batch_end")  # 批次结束回调
    
            # 周期结束处理
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 记录学习率
            self.run_callbacks("on_train_epoch_end")  # 周期结束回调
            
            # 主进程操作
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                # 更新EMA模型参数
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
    
                # 验证阶段
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()  # 执行验证
                # 保存指标
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                # 更新停止条件
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
    
                # 模型保存
                if self.args.save or final_epoch:
                    self.save_model()  # 保存模型
                    self.run_callbacks("on_model_save")  # 模型保存回调
    
            # 调度器相关
            t = time.time()
            self.epoch_time = t - self.epoch_time_start  # 计算epoch耗时
            self.epoch_time_start = t
            # 时间模式下的epoch数调整
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()  # 重新设置调度器
                self.scheduler.last_epoch = self.epoch  # 同步调度器epoch
                self.stop |= epoch >= self.epochs  # 检查是否超过总epoch数
            self.run_callbacks("on_fit_epoch_end")  # 训练周期结束回调
            self._clear_memory()  # 清理内存
    
            # 分布式训练同步停止状态
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:  # 满足停止条件则跳出循环

                ######################——————自定义——————#####################################################################################################
                if attack_task=='fgsm':
    
                    print("保存结果图")
                    def fgsm_attack(image, epsilon, data_grad):
                        # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
                        sign_data_grad = data_grad.sign()
                        # 通过epsilon生成对抗样本
                        perturbed_image = image + epsilon*sign_data_grad
                        # # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
                        perturbed_image = torch.clamp(perturbed_image, 0, 1)
                        # 返回对抗样本
                        return perturbed_image
                    def show_image(images):
                        images=all_imgs[0].detach().numpy()#shape(batch_size,3,640,640)
                        fig,axes=plt.subplots(1,5,figsize=(20,4))
                        for i,ax in enumerate(axes):
                            ax.imshow(images[i].transpose(1,2,0))
                            ax.axis('off')
                        plt.show()

            
                    #grads[0].shape(batch_size,3,640,640)
                    for i in range(len(all_imgs)):
                        images=all_imgs[i].clone()
                        img_names=img_path[i]
                        if not os.path.exists(save_adv_path):
                            os.makedirs(save_adv_path)
                        for eps in epsilons:
                            target_path=""
                            if not os.path.exists(os.path.join(save_adv_path,f"eps_{eps}")):
                                os.makedirs(os.path.join(save_adv_path,f"eps_{eps}"))
                                target_path=os.path.join(save_adv_path,f"eps_{eps}")
                            elif os.path.exists(os.path.join(save_adv_path,f"eps_{eps}")):
                                target_path=os.path.join(save_adv_path,f"eps_{eps}")
                                
                            for index in range(len(images)):
                                
                                image=images[index]
                                img_name=os.path.basename(img_names[index])

                                # img_min_value = image.min().item()  # 获取最小值
                                # img_max_value = image.max().item()  # 获取最大值
                                # grad_min_value = grads[i][0][index].min().item()  # 获取最小值
                                # grad_max_value = grads[i][0][index].max().item()  # 获取最大值
                                #image范围[0,1] grads范围[-1,1]
        
                                # image=image.permute(1,2,0).cpu().numpy()
                                # plt.imshow(image)
                                # plt.show()


                                adv_image = fgsm_attack(image=image, epsilon=eps,data_grad= grads[i][0][index])
                                adv_image = adv_image.permute(1,2,0).detach().cpu().numpy()  #adv_image.shape(3,640,640)
                                plt.imsave(os.path.join(target_path,f"{img_name}"),adv_image)
                    print('保存')

                #############################################################################################################################################################
                break
            epoch += 1  # 更新epoch计数

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Get batch size by calculating memory occupation of model."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / (2**30)

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "item"}
            self.data["nc"] = 1
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """执行训练优化器的单步操作，包括梯度裁剪和EMA更新。"""
        self.scaler.unscale_(self.optimizer)  # 反缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 裁剪梯度
        self.scaler.step(self.optimizer)  # 执行优化步骤
        self.scaler.update()  # 更新缩放器
        self.optimizer.zero_grad()  # 清空梯度
        if self.ema:
            self.ema.update(self.model)  # 更新EMA模型

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args


                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0


        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
