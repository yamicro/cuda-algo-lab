# cuda-algo-lab

CUDA vs Python 算法性能对比练习项目

---

## 环境信息

* GPU: RTX 4080S (16GB)
* CUDA: 12.8
* Python: 推荐 3.10+
* 依赖: CMake, pybind11, numpy, torch

---

## 构建指南

```bash
mkdir build && cd build
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make -j
```

## 性能测试方式

项目中各算法都对应一个 benchmark script，位于 `benchmark/` 目录下，直接运行即可：

```bash
python benchmark/bench_relu.py
```

想一次性并行测试所有算法应该运行总测试脚本：

```bash
python benchmark/benchmark_util.py
```


**注意**：结果取出平均耗时，并在 CUDA 实现前有输入处理和重复运行预热等调整

---

## 项目结构总览

```
cuda-algo-lab/
├── benchmark/      # benchmark 每个算法的测试脚本
├── build/          # cmake build 目录（不需提交项目）
├── cmake/          # cmake 关联脚本
├── src/            # CUDA、C++、pybind 源文件
├── python/         # python 实现 baseline
├── triton/         # triton 代码
├── third_party/    # pybind11 代码
├── setup.py        # 配合 python 安装模式编译
└── README.md
```

---

## 算法运行时间对比表

单位：毫秒（ms）

| Algorithm  | CUDA       | CUDA for 4 pack | Python       |
| ---------- | ---------- | --------------- | ------------ |
| add        | 0.00018880 | 0.00016320      | 1.17540359   |
| elu        | 0.00016320 | 0.00013120      | 10.90288162  |
| embedding  | 0.00062720 | 0.00014400      | 59.88526344  |
| gelu       | 0.00016640 | 0.00015040      | 19.69265938  |
| hist       | 0.00014080 | -               | 698.50611687 |
| relu       | 0.00020800 | 0.00015680      | 1.13320351   |
| sigmoid    | 0.00018240 | 0.00012800      | 3.19385529   |
| swish      | 0.00018880 | 0.00015360      | 4.29010391   |
| transpose  | 0.00014080 | 0.00016000      | 206.79092407 |
| warpreduce | 0.00055360 | 0.00020480      | 0.18358231   |

---

## Triton vs cuBLAS 对比（GEMM）

单位：毫秒（ms），矩阵大小：4096x4096

| 算法类型   | cuBLAS (ms) | Triton (ms) | 
| ------ | ----------- | ----------- | 
| matmul | 90.048791   | 91.990989   | 
