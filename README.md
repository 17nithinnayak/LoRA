# LoRA Implementation from Scratch 🚀

A clean, educational implementation of **Low-Rank Adaptation (LoRA)** in PyTorch.

## 📖 What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to adapt large pre-trained models (like GPT-3, LLaMA, or Stable Diffusion) using a **fraction of the parameters**.

### The Problem LoRA Solves

Fine-tuning a model like GPT-3 (175B parameters) traditionally requires:
- ❌ Updating all 175 billion parameters
- ❌ Storing full gradients and optimizer states
- ❌ Massive GPU memory (hundreds of GBs)
- ❌ Hours or days of training time

### The LoRA Solution

LoRA introduces small "adapter" matrices that:
- ✅ Keep original weights **frozen** (no updates)
- ✅ Train only **0.1-1%** of the parameters
- ✅ Reduce memory usage by **3x** or more
- ✅ Train **10-100x faster**
- ✅ Maintain similar performance to full fine-tuning

---

## 🧠 How LoRA Works

Instead of updating a weight matrix **W** (shape `[d, d]`), LoRA adds a low-rank decomposition:
```
W' = W + ΔW
ΔW = B × A
```

Where:
- **W**: Original frozen weights `[d_out, d_in]`
- **A**: Low-rank matrix `[rank, d_in]` (randomly initialized)
- **B**: Low-rank matrix `[d_out, rank]` (initialized to zeros)
- **rank**: Typically 4-64 (much smaller than d)

### The Magic

For a layer with 512 → 512 dimensions:
- **Full fine-tuning**: 512 × 512 = **262,144 parameters**
- **LoRA (rank=8)**: (512 × 8) + (8 × 512) = **8,192 parameters** (97% reduction!)

---

## 🛠️ Code Structure

### `LoRALayer`
The core adapter module that implements the low-rank matrices.
```python
LoRALayer(in_dim=512, out_dim=256, rank=4, alpha=8)
```

**Key Features:**
- Matrix **A**: Kaiming-initialized for stable gradients
- Matrix **B**: Zero-initialized (so initial ΔW = 0)
- Scaling factor: `alpha / rank` for consistent learning rates

### `LinearWithLoRA`
Wraps an existing `nn.Linear` layer with LoRA adaptation.
```python
original_layer = nn.Linear(512, 256)
lora_layer = LinearWithLoRA(original_layer, rank=4, alpha=8)
```

**Key Features:**
- Freezes original layer weights
- Adds trainable LoRA adapter
- Forward pass: `output = original(x) + adapter(x)`

---

**Output:**
```
Original: tensor([[ 0.5747, -1.0012, -0.4544, -0.2457, -0.0885]])
LoRA    : tensor([[ 0.5747, -1.0012, -0.4544, -0.2457, -0.0885]])
Match?  : True

Total Params: 85
Trainable:    30 (Only 35.3%!)
```

---

## 🎯 Key Hyperparameters

### `rank` (r)
- **Range**: 1-256 (typically 4-64)
- **Effect**: Higher rank = more expressiveness but more parameters

### `alpha` (α)
- **Range**: Equal to or 2x the rank
- **Effect**: Controls the magnitude of LoRA updates
- **Formula**: Effective learning rate = `(alpha / rank) * lr`

---

## 🔬 Why B is Initialized to Zero

This is **crucial** for LoRA:
```
self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
```

**Reason:** At initialization, `ΔW = B @ A = 0`, so:
```
output = W(x) + ΔW(x) = W(x) + 0 = W(x)
```

The model behaves **exactly** like the original frozen model. As training progresses, B learns to add useful corrections.

---

## 📚 Applications

LoRA has been successfully used for:

- 🤖 **Language Models**: GPT-3, LLaMA, Mistral fine-tuning
- 🎨 **Image Generation**: Stable Diffusion custom styles
- 🗣️ **Speech Models**: Whisper adaptation
- 🧬 **Scientific Models**: Protein folding, drug discovery

---

## 🔗 References

- **Original Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- **Hugging Face PEFT**: [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **Microsoft LoRA**: [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

---
