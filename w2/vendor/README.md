The engine source is upstream [PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow) at commit `7ef897eaedc16626a281ccfe0d453fae80251e8f`.

From the repository root:

```bash
git clone https://github.com/PanZaifeng/KVFlow.git w2/vendor/KVFlow
git -C w2/vendor/KVFlow checkout 7ef897eaedc16626a281ccfe0d453fae80251e8f
git -C w2/vendor/KVFlow apply ../../patches/w2-combined.patch
```

`w2/patches/w2-combined.patch` is the two cache fixes against that commit.
