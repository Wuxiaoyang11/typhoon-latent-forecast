from typing import List

# 一个简单的调度器类，用于在训练过程中动态调整某个值
class SimpleScheduler:
    def __init__(self,
                 value_range: List,  # 值的范围，一个包含起始值和结束值的列表
                 warmup: int,  # 预热步数，在此步数之前，值保持不变
                 last: int,  # 最后的步数，在此步数之后，值保持不变
                 verbose=False):  # 是否打印值的变化
        self.value_range = value_range
        self.warmup = warmup
        self.last  = last

        self.current_step = 0  # 当前步数
        self.current_value = value_range[0]  # 当前值
        self.range = value_range[1] - value_range[0]  # 值的范围大小
        self.step_range = self.last - self.warmup  # 步数范围大小
        self.verbose = verbose

    # 更新当前步数和值
    def step(self):
        self.current_step += 1
        # 如果当前步数在预热期之前或在最后一步之后，则不更新值
        if self.current_step < self.warmup or self.current_step > self.last:
            return

        # 根据当前步数在步数范围内的比例，线性地更新值
        prop = (self.current_step - self.warmup) / self.step_range
        old_value = self.current_value
        self.current_value = prop * self.range + self.value_range[0]

        # 如果值的范围是整数，则将当前值四舍五入为整数
        if isinstance(self.value_range[0], int):
            self.current_value = round(self.current_value)

        # 如果启用了详细模式并且值发生了变化，则打印新值
        if self.verbose and self.current_value != old_value:
            print(f"New Value: {self.current_value}")

    # 获取当前值
    def get_value(self):
        return self.current_value
