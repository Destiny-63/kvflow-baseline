引擎源码使用上游 [PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow) 的 commit `7ef897eaedc16626a281ccfe0d453fae80251e8f`。

在仓库根目录执行：

```bash
git clone https://github.com/PanZaifeng/KVFlow.git w2/vendor/KVFlow
git -C w2/vendor/KVFlow checkout 7ef897eaedc16626a281ccfe0d453fae80251e8f
git -C w2/vendor/KVFlow apply ../../patches/w2-combined.patch
```

`w2/patches/w2-combined.patch` 是相对该 commit 的两处缓存修复。
