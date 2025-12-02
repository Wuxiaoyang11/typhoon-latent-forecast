import numpy as np
import torch
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from torch.utils.data import Dataset

# 一个包装类，用于将PyTorch的Subset对象转换为一个完整的数据集对象
# 这样可以方便地对数据子集应用变换（transform）
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset  # 原始的数据子集
        self.transform = transform  # 需要应用的数据变换

    def __getitem__(self, index):
        # 获取子集中的原始数据
        data = self.subset[index]
        # 如果定义了变换，则对数据进行变换
        if self.transform:
            return self.transform(data)
        return data

    def __len__(self):
        # 返回子集的大小
        return len(self.subset)

# 一个用于生成时间序列数据对的数据集类
# 主要用于自监督学习，生成时间上邻近的图像对
class TemporalSequencePairDataset(Dataset):
    def __init__(self, images, labels, transform) -> None:
        self.images = images  # 图像列表
        self.labels = labels  # 标签列表
        self.transform = transform  # 数据变换

    def __getitem__(self, index):
        size = len(self.images)
        # 随机选择第二个图像的索引，使其在第一个图像之后的一个小时间窗口内（最多6个时间步）
        # 使用beta分布使得更近的图像对被选中的概率更高
        index2 = int(np.random.beta(2, 5) * np.min([6, (size - index)])) + index

        x1, lbl1 = self.images[index], self.labels[index]
        x2, lbl2 = self.images[index2], self.labels[index2]

        # 对图像和标签应用变换
        img1, dt1 = self.transform((x1, lbl1))
        img2, dt2 = self.transform((x2, lbl2))

        # 返回两个变换后的图像，以及它们之间的时间差（以小时为单位）
        return img1, img2, (dt2-dt1).total_seconds()/3600

    def __len__(self):
        # 返回图像序列的长度
        return len(self.images)


# 一个嵌套的数据集类，继承自DigitalTyphoonDataset
# 数据集中的每个样本本身是另一个数据集（TemporalSequencePairDataset）
class NestedDigitalTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 split_dataset_by="image",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(image_dir, metadata_dir, metadata_json, labels, split_dataset_by, spectrum, True, load_data_into_memory, ignore_list, filter_func, transform_func, transform, verbose)

    def __getitem__(self, idx):
        # 获取第idx个台风序列
        seq = self.get_ith_sequence(idx)
        # 获取序列中的所有图像
        images = seq.get_all_images_in_sequence()
        image_arrays = np.array([image.image() for image in images])
        labels = np.array([self._labels_from_label_strs(image, self.labels) for image in images])

        # 返回一个包含该序列所有图像和标签的TemporalSequencePairDataset实例
        return TemporalSequencePairDataset(image_arrays, labels, self.transform)

# 为MoCo（Momentum Contrast）自监督学习设计的数据集类
class MoCoSequenceDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 window_size_scheduler,  # 一个用于动态调整时间窗口大小的调度器
                 split_dataset_by="image",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(image_dir,
                         metadata_dir,
                         metadata_json,
                         labels,
                         split_dataset_by,
                         spectrum,
                         True,
                         load_data_into_memory,
                         ignore_list,
                         filter_func,
                         transform_func,
                         transform,
                         verbose)
        self.window_size_scheduler = window_size_scheduler

    def __getitem__(self, idx):
        # 获取第idx个台风序列
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()

        # 从调度器获取当前的最大时间窗口
        max_range = self.window_size_scheduler.get_value()

        # 随机选择序列中的第一张图像
        idx1 = np.random.randint(0, len(images))
        # 在时间窗口内随机选择第二张图像，作为正样本对
        idx2 = np.min([idx1 + np.random.randint(0, max_range), len(images) - 1])

        # 对两张图像应用变换并返回
        return self.transform(images[idx1].image()), self.transform(images[idx2].image())

# 定义用于独热编码的类别标签的大小
LABEL_SIZE = dict(
    month=12,
    day=31,
    hour=24,
)

# 定义用于标准化的连续标签的均值和标准差
NORMALIZATION = dict(
    pressure=[983.8, 22.5],  # 气压
    wind=[36.7, 32.7],      # 风速
    lat=[22.58, 10.6],      # 纬度
    lng=[136.2, 17.3],      # 经度
)

# 用于时间序列预测任务的数据集类
class SequenceTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 labels,  # 需要包含的标签列表
                 x,  # 用作输入的特征索引
                 y,  # 用作目标的特征索引
                 num_inputs,  # 输入序列的长度
                 num_preds,  # 预测序列的长度
                 interval=1,  # 序列中数据点之间的时间间隔
                 output_all=False,  # 是否输出整个序列
                 preprocessed_path=None,  # 预处理图像特征的路径
                 latent_dim=None,  # 图像特征的维度
                 pred_diff=False,  # 是否预测与最后一个输入值的差值
                 prefix="/fs9/datasets/typhoon-202404/wnp",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(f"{prefix}/image/",
                        f"{prefix}/metadata/",
                        f"{prefix}/metadata.json",
                         labels,
                         "sequence",
                         spectrum,
                         True,
                         load_data_into_memory,
                         ignore_list,
                         filter_func,
                         transform_func,
                         transform,
                         verbose)
        idx = 0
        self.x = []  # 输入特征的索引列表
        self.y = []  # 输出特征的索引列表

        # 根据labels和x, y参数，构建输入和输出特征的索引
        for i, label in enumerate(labels):
            sz = LABEL_SIZE[label] if label in LABEL_SIZE else 1
            if i in x:
                self.x.extend(list(range(idx, idx+sz)))
            if i in y:
                self.y.extend(list(range(idx, idx+sz)))
            idx += sz

        # 如果使用预处理的图像特征
        if preprocessed_path is not None:
            assert latent_dim is not None
            # -1作为图像特征的特殊索引
            if -1 in x:
                self.x.extend(list(range(idx, idx+latent_dim)))
            if -1 in y:
                self.y.extend(list(range(idx, idx+latent_dim)))

        print(f"\n模型输入维度: {self.get_input_size()}")
        print(f"模型输出维度: {self.get_output_size()}\n")

        self.preprocessed_path = preprocessed_path
        self.num_inputs = num_inputs
        self.num_preds = num_preds
        self.interval = interval
        self.output_all = output_all
        self.pred_diff = pred_diff

        # 定义用于切分输入和输出序列的lambda函数
        self.slice_inputs = lambda start_idx: slice(start_idx, start_idx+(self.num_inputs*self.interval),self.interval)
        self.slice_outputs = lambda start_idx: slice(start_idx+(self.num_inputs*self.interval),start_idx+((self.num_inputs+self.num_preds)*self.interval), self.interval)

        # 后处理步骤：过滤掉太短的序列
        def filter_sequences(sequence):
            if sequence.get_num_images() < (self.num_inputs + self.num_preds)*self.interval+1 or sequence.images[0].year() < 1987:
                return True
            return False

        # 遍历所有序列并应用过滤器
        for seq in self.sequences:
            if filter_sequences(seq):
                self.number_of_images -= seq.get_num_images()
                seq.images.clear()
                self.number_of_nonempty_sequences -= 1
        self.number_of_nonempty_sequences += 1

    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()

        # 从图像中提取标签并堆叠成张量
        labels = torch.stack([self._labels_from_label_strs(image, self.labels) for image in images])

        # 如果提供了预处理特征的路径，则加载它们
        if self.preprocessed_path is not None:
            npz = np.load(f"{self.preprocessed_path}/{seq.sequence_str}.npz")
            names_to_features = dict(zip(npz["arr_1"], npz["arr_0"]))
            features = [names_to_features[str(img.image_filepath).split("/")[-1].split(".")[0]]
                        for img in images]
            features = torch.from_numpy(np.array(features))
            # 将图像特征与标签拼接
            labels = torch.cat((labels, features), dim=1)

        if self.output_all:
            return labels, seq.sequence_str

        # 随机选择一个起始点，以创建输入和输出子序列
        start_idx = np.random.randint(0, seq.get_num_images()-(self.num_inputs + self.num_preds)*self.interval)
        lab_inputs = labels[self.slice_inputs(start_idx)][:, self.x]
        lab_preds = labels[self.slice_outputs(start_idx)][:, self.y]

        # 如果pred_diff为True，则预测与最后一个输入值的差值
        if self.pred_diff:
            last_input_val = labels[self.slice_inputs(start_idx)][-1][self.y]
            lab_preds = lab_preds - last_input_val

        return lab_inputs, lab_preds

    def get_input_size(self):
        # 获取输入特征的总维度
        return len(self.x)

    def get_output_size(self):
        # 获取输出特征的总维度
        return len(self.y)

    def _labels_from_label_strs(self, image, label_strs):
        # 从图像对象中根据标签字符串提取标签值
        if isinstance(label_strs, list) or isinstance(label_strs, tuple):
            label_ray = torch.cat([self._prepare_labels(image.value_from_string(label), label) for label in label_strs])
            return label_ray
        else:
            label = self._prepare_labels(image.value_from_string(label_strs), label_strs)
            return label

    def _prepare_labels(self, value, label):
        # 准备标签：对分类标签进行独热编码，对连续标签进行归一化
        if label in LABEL_SIZE:
            one_hot = torch.zeros(LABEL_SIZE[label])
            if label == "hour":
                one_hot[value] = 1
            elif label == "grade":
                one_hot[value-2] = 1
            else:
                one_hot[value-1] = 1
            return one_hot
        else:
            # 归一化
            if label in NORMALIZATION:
                mean, std = NORMALIZATION[label]
                return (torch.Tensor([value]) - mean) / std
            return torch.Tensor([value])

    def get_sequence_images(self, seq_str):
        # 获取指定序列的所有图像（经过裁剪）
        def crop(img, cropx=224, cropy=224):
            y,x = img.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            return img[starty:starty+cropy,startx:startx+cropx]
        idx = self._sequence_str_to_seq_idx[seq_str]
        seq = self.sequences[idx]
        images = seq.get_all_images_in_sequence()
        return [crop(image.image()) for image in images]

    def get_sequence(self, seq_str):
        # 获取指定序列的完整处理后数据
        idx = self._sequence_str_to_seq_idx[seq_str]
        return self.__getitem__(idx)
