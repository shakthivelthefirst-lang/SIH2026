# Hardware-Accelerated Speech Enhancement Pipeline: ML-to-RTL Interface Specification

**Document Identifier:** `SPEC-ML-RTL-001`  
**Target Platform:** AMD/Xilinx 7-Series / UltraScale+ FPGA (Vivado Design Suite)  
**Author:** Embedded ML / FPGA Systems Engineering  
**Status:** Approved / Baseline Contract  

---

## 1. System Overview & Architectural Pipeline

This specification defines the strict interface, mathematical representation, quantization budget, memory organization, and signaling protocol between the machine learning model (trained in PyTorch/TensorFlow) and the synthesized RTL core (implemented in SystemVerilog/VHDL for Vivado).

The target system is a low-latency, real-time speech enhancement accelerator that computes a 16-channel spectral suppression mask from the low-frequency magnitude spectrum of incoming 16 kHz audio.

```
       16 kHz Audio In
              │
              ▼
   ┌──────────────────────┐
   │  Framing & Window    │  512-sample buffer, 256-sample hop, Hann window
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │     512-pt FFT       │  Real-to-Complex (257 unique positive frequency bins)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Magnitude Extraction │  Bins k = 0..31 (|X[k]| = sqrt(Re^2 + Im^2))
   └──────────┬───────────┘
              │
              ▼ [AXI4-Stream INT8 Features: 32 bins]
   ╔══════════════════════════════════════════════════════════════════════╗
   ║               FPGA ML ACCELERATOR CORE (MLP IP)                      ║
   ║                                                                      ║
   ║   Dense Layer 1:  32 in  ──► 32 out  [Weights INT8, Bias INT32]      ║
   ║   Activation:     ReLU                                               ║
   ║   Dense Layer 2:  32 in  ──► 16 out  [Weights INT8, Bias INT32]      ║
   ║   Activation:     Sigmoid / Mask Scaling Approximation               ║
   ╚══════════════════════════════════════════════════════════════════════╝
              │
              ▼ [AXI4-Stream INT8 Mask: 16 values]
   ┌──────────────────────┐
   │  Spectral Masking    │  Bins k = 0..15: X_enh[k] = X[k] * (Mask[k] / 255)
   │                      │  Bins k = 16..256: Unmodified / High-band policy
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │     512-pt IFFT      │  Complex-to-Real inverse transform
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │   Overlap-Add (OLA)  │  Reconstructed time-domain enhanced audio
   └──────────────────────┘
```

---

## 2. Frame & Signal Parameters (Audio DSP Contract)

The front-end DSP module must feed the ML core in lockstep with the framing parameters below:

| Parameter | Specification | Hardware Notes / Constraints |
| :--- | :--- | :--- |
| **Audio Sampling Rate ($f_s$)** | 16,000 Hz (16 kHz, mono) | Audio sample period $T_s = 62.5\text{ }\mu\text{s}$. Single channel 16-bit PCM. |
| **Frame Length ($N_{frame}$)** | 512 samples (32.0 ms) | Input buffer: Dual ping-pong BRAM (512 $\times$ 16-bit). |
| **Hop Length ($N_{hop}$)** | 256 samples (16.0 ms) | 50% frame overlap. Frame stride interval $\Delta T = 16\text{ ms}$. |
| **Window Function** | Periodic Hann Window | Pre-computed, stored in 512-entry ROM as Q1.15 signed fixed-point. |
| **FFT Resolution ($N_{fft}$)** | 512-point R2C FFT | Radix-2 / Radix-4 pipeline FFT producing 257 complex bins ($k=0 \dots 256$). |
| **Bin Frequency Resolution ($\Delta f$)** | $31.25\text{ Hz / bin}$ | $\Delta f = f_s / N_{fft} = 16000 / 512 = 31.25\text{ Hz}$. |
| **Frame Arrival Budget** | 16.0 ms | At 100 MHz sys_clk, $1.6 \times 10^6$ clock cycles are available per frame. |

### Hann Window Mathematical Definition
$$w[n] = 0.5 \cdot \left(1 - \cos\left(\frac{2\pi n}{N_{frame}}\right)\right), \quad 0 \le n < 512$$
*Quantization:* Fixed-point Q1.15 format, mapped to 16-bit signed integer:
$$w_{q}[n] = \text{round}\left(w[n] \cdot 32767\right)$$

---

## 3. Input Feature Space & Quantization Contract

### 3.1 Feature Slicing
* **Extracted Bins:** First 32 magnitude bins, $k \in [0, 31]$.
* **Spectral Bandwidth:** 
  $$f_{min} = 0.0\text{ Hz (DC)}, \quad f_{max} = 31 \times 31.25\text{ Hz} = 968.75\text{ Hz}$$
  Covers pitch fundamental frequencies ($F_0 \approx 85\text{--}255\text{ Hz}$), first formant resonances ($F_1$), and severe low-frequency industrial/wind noise.
* **Magnitude Computation:**
  $$|X[k]| = \sqrt{\text{Re}\{X[k]\}^2 + \text{Im}\{X[k]\}^2}$$
  Hardware implementation via 16-iteration pipelined CORDIC engine or Alpha-Max Beta-Min approximation.

### 3.2 Fixed-Point Representation & Scaling
To ensure maximum dynamic range and direct compatibility with DSP48 slices:

| Feature Dimension | Fixed-Point Format | Numerical Range | Binary Representation |
| :--- | :--- | :--- | :--- |
| **Magnitude Input $X_{in}[k]$** | Unsigned Q0.8 (`uint8_t`) | $[0.0, 0.99609375]$ (normalized) | 8-bit unsigned `[7:0]` ($0 \dots 255$) |
| **Alternative Representation** | Signed Q0.7 (`int8_t`) | $[0, 127]$ (positive half-range) | 8-bit signed 2's complement `[7:0]` |

* **Standard Hardware Baseline:** **Unsigned INT8 (`uint8_t`, Q0.8)**.  
  Zero is represented by `0x00` (silence), and clipping ceiling is `0xFF` (peak spectral power).
* **Scaling Factor ($S_{in}$):**
  $$X_{quant}[k] = \min\left(\left\lfloor \frac{|X[k]|}{S_{in}} \right\rceil, 255\right)$$
  Where $S_{in} = \text{Dynamic\_Peak} / 255.0$.

---

## 4. MLP Architecture Specification

The network is a high-efficiency 2-layer Multilayer Perceptron (MLP) mapping 32 input spectral magnitudes to 16 attenuation coefficients.

```
       Input Layer               Hidden Layer               Output Layer
       (32 Features)             (32 Neurons)               (16 Neurons)
       
       x[0]  ─────┬────────────► h[0]  ─────┬─────────────► Mask[0] (Bin 0)
       x[1]  ─────┼────────────► h[1]  ─────┼─────────────► Mask[1] (Bin 1)
         :        │                :        │                 :
       x[31] ─────┴────────────► h[31] ─────┴─────────────► Mask[15] (Bin 15)
       
                    W1: 32x32                  W2: 16x32
                    B1: 32                     B2: 16
                    Activation: ReLU           Activation: Sigmoid LUT
```

### 4.1 Layer 1: Dense + ReLU
* **Input Vector:** $\mathbf{x} \in \mathbb{R}^{32}$, quantized as unsigned 8-bit integers (`uint8_t`).
* **Weights Matrix ($W_1$):** $32 \times 32 = 1024$ parameters, signed 8-bit (`int8_t`, Q1.7, range $[-128, 127]$).
* **Bias Vector ($\mathbf{b}_1$):** 32 parameters, signed 32-bit (`int32_t`).
* **Accumulator:** 32-bit signed integer.
* **Affine Transform:**
  $$z_{1}[j] = \sum_{i=0}^{31} \left( W_{1}[j][i] \times x[i] \right) + b_{1}[j], \quad j \in [0, 31]$$
* **Activation Function:** Rectified Linear Unit (ReLU) + Requantization:
  $$h[j] = \text{ReLU}(z_{1}[j]) = \begin{cases} z_{1}[j], & z_{1}[j] > 0 \\ 0, & z_{1}[j] \le 0 \end{cases}$$
* **Downscaling & Requantization to INT8 (Q0.8):**
  $$a_{1}[j] = \min\left( \max\left( \left\lfloor z_{1}[j] \gg \text{SHIFT}_1 \right\rfloor, 0 \right), 255 \right)$$
  Where $\text{SHIFT}_1$ is determined during post-training quantization calibration (typically $7 \dots 10$ bits).

### 4.2 Layer 2: Dense + Sigmoid
* **Input Vector:** $\mathbf{a}_1 \in \mathbb{R}^{32}$, quantized as unsigned 8-bit integers (`uint8_t`).
* **Weights Matrix ($W_2$):** $16 \times 32 = 512$ parameters, signed 8-bit (`int8_t`, Q1.7, range $[-128, 127]$).
* **Bias Vector ($\mathbf{b}_2$):** 16 parameters, signed 32-bit (`int32_t`).
* **Affine Transform:**
  $$z_{2}[k] = \sum_{j=0}^{31} \left( W_{2}[k][j] \times a_{1}[j] \right) + b_{2}[k], \quad k \in [0, 15]$$
* **Activation Function:** Fixed-point Sigmoid $\sigma(z)$ producing output mask:
  $$M[k] = \sigma(z_2[k]) = \frac{1}{1 + e^{-z_2[k]}}$$

### 4.3 Sigmoid Hardware Implementation Contract
Hardware engines shall implement Sigmoid via one of two synthesis options:

* **Option A: 256-Word BRAM / Distributed ROM Lookup Table (Preferred):**
  Address index computed by saturating $z_{2}[k] \gg \text{SHIFT}_2$ into an 8-bit signed index $[-128, 127]$ (mapped to `0x00..0xFF`). Output directly returns `uint8_t` in $[0, 255]$ ($0 = 0.0$ attenuation, $255 = 1.0$ full transmission).
* **Option B: Piecewise Linear Approximation (Hard-Sigmoid):**
  $$M_{approx}(z) = \text{clip}\left(\left\lfloor \frac{z_{scaled} + 128}{2} \right\rfloor, 0, 255\right)$$

### 4.4 Output Mask Application Policy
* **Bins $k = 0 \dots 15$:** Attenuation mask applied directly:
  $$\hat{X}_{Re}[k] = \left\lfloor \frac{X_{Re}[k] \times M[k]}{255} \right\rfloor, \quad \hat{X}_{Im}[k] = \left\lfloor \frac{X_{Im}[k] \times M[k]}{255} \right\rfloor$$
* **Bins $k = 16 \dots 256$:** Pass-through unmodified ($M[k] = 1.0$) or attenuated by static global parameter to preserve high-frequency air/fricatives while eliminating vocal band noise.

---

## 5. Hardware Quantization Budget & Arithmetic Guarantees

| Tensor | Precision | Format | Dynamic Range | Hardware Primitive |
| :--- | :--- | :--- | :--- | :--- |
| **Input Features ($\mathbf{x}$)** | 8-bit unsigned | Q0.8 | $[0, 255]$ | BRAM / Distributed RAM |
| **Layer 1 Weights ($W_1$)** | 8-bit signed | Q1.7 (2's comp) | $[-128, 127]$ | ROM / Distributed RAM |
| **Layer 1 Biases ($\mathbf{b}_1$)** | 32-bit signed | Q9.23 (2's comp) | $[-2^{31}, 2^{31}-1]$ | ROM / Distributed RAM |
| **Layer 1 Accumulator** | 32-bit signed | 2's complement | $[-2^{31}, 2^{31}-1]$ | DSP48 Slice Accumulator |
| **Layer 1 Outputs ($\mathbf{a}_1$)**| 8-bit unsigned | Q0.8 | $[0, 255]$ | Pipeline Register |
| **Layer 2 Weights ($W_2$)** | 8-bit signed | Q1.7 (2's comp) | $[-128, 127]$ | ROM / Distributed RAM |
| **Layer 2 Biases ($\mathbf{b}_2$)** | 32-bit signed | Q9.23 (2's comp) | $[-2^{31}, 2^{31}-1]$ | ROM / Distributed RAM |
| **Layer 2 Accumulator** | 32-bit signed | 2's complement | $[-2^{31}, 2^{31}-1]$ | DSP48 Slice Accumulator |
| **Output Mask ($\mathbf{M}$)** | 8-bit unsigned | Q0.8 | $[0, 255]$ ($0.0 \dots 1.0$) | Output Streaming FIFO |

### 5.1 Accumulator Overflow Margin Proof
For any neuron computing a 32-input dot product:
$$\text{Max Possible Negative Acc} = 32 \times (255 \times -128) = 32 \times (-32640) = -1,044,480$$
$$\text{Max Possible Positive Acc} = 32 \times (255 \times 127) = 32 \times (32385) = +1,036,320$$
With 32-bit signed accumulator range:
$$\text{Capacity} = [-2,147,483,648, \quad +2,147,483,647]$$
* **Headroom:** Margin exceeds **11 Guard Bits**. Overflows during MAC operations are mathematically impossible regardless of input vectors.

### 5.2 AMD/Xilinx DSP48 Mapping Guarantee
* **DSP48E1 / DSP48E2 Architecture:** Supports $25 \times 18$ (or $27 \times 18$) signed multiplication with a 48-bit accumulator.
* **Mapping:**
  * Port A: 8-bit unsigned activation zero-extended to 9 bits signed.
  * Port B: 8-bit signed weight sign-extended to 18 bits.
  * Port C / P: 32-bit bias and 48-bit internal accumulation.
  * Each MAC executes in a single clock cycle at clock rates up to 450 MHz.

---

## 6. File Format Contract (Verilog `$readmemh` Specifications)

All model parameter files must be exported as pure ASCII hexadecimal files (`.mem`) compliant with IEEE 1364 Verilog `$readmemh`.

### 6.1 Formatting Rules
1. No `0x` prefixes.
2. One value per line.
3. Negative values must be formatted as 2's complement hexadecimal of the exact bit width:
   * 8-bit: Two hex digits (e.g., `-1` $\to$ `FF`, `-128` $\to$ `80`, `+127` $\to$ `7F`).
   * 32-bit: Eight hex digits (e.g., `-1` $\to$ `FFFFFFFF`, `+300` $\to$ `0000012C`).
4. Optional inline comments prefixed with `//`.
5. Address pointers (`@address`) are allowed at segment boundaries.

### 6.2 File Manifest & Memory Layout

| File Name | Dimensions | Data Type | Hex Width | Depth (Lines) | Organization / Ordering |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `w1.mem` | $32 \times 32$ | `int8_t` | 2 hex chars | 1024 | Row-major: `W1[neuron 0][in 0..31]`, `W1[neuron 1]...` |
| `b1.mem` | 32 | `int32_t` | 8 hex chars | 32 | Sequential: `B1[0]`, `B1[1]`, ..., `B1[31]` |
| `w2.mem` | $16 \times 32$ | `int8_t` | 2 hex chars | 512 | Row-major: `W2[neuron 0][in 0..31]`, `W2[neuron 1]...` |
| `b2.mem` | 16 | `int32_t` | 8 hex chars | 16 | Sequential: `B2[0]`, `B2[1]`, ..., `B2[15]` |
| `lut_sigmoid.mem` | 256 | `uint8_t` | 2 hex chars | 256 | Indexed by 8-bit address `00` to `FF` |

### 6.3 Verilog Memory Instantiation Reference
```verilog
// SystemVerilog ROM Instantiation Example
logic signed [7:0]  w1_rom [0:1023];
logic signed [31:0] b1_rom [0:31];
logic signed [7:0]  w2_rom [0:511];
logic signed [31:0] b2_rom [0:15];
logic [7:0]         sigmoid_rom [0:255];

initial begin
    $readmemh("w1.mem", w1_rom);
    $readmemh("b1.mem", b1_rom);
    $readmemh("w2.mem", w2_rom);
    $readmemh("b2.mem", b2_rom);
    $readmemh("lut_sigmoid.mem", sigmoid_rom);
end
```

---

## 7. RTL Interface Contract & Signal Protocol

The ML accelerator core exposes standard AXI4-Stream slave and master interfaces.

```
                  ┌─────────────────────────────────────────┐
                  │        Speech Enhancement MLP Core       │
                  │                                         │
   clk ──────────►│ aclk                                    │
   rst_n ────────►│ aresetn                                 │
                  │                                         │
                  │ [AXI4-Stream Slave: Input Features]     │
   s_axis_tdata ─►│ s_axis_tdata [7:0]                      │
   s_axis_tvalid─►│ s_axis_tvalid                           │
   s_axis_tready◄─│ s_axis_tready                           │
   s_axis_tlast ─►│ s_axis_tlast                            │
                  │                                         │
                  │ [AXI4-Stream Master: Output Mask]       │
   m_axis_tdata ◄─│ m_axis_tdata [7:0]                      │
   m_axis_tvalid◄─│ m_axis_tvalid                           │
   m_axis_tready─►│ m_axis_tready                           │
   m_axis_tlast ◄─│ m_axis_tlast                            │
                  └─────────────────────────────────────────┘
```

### 7.1 Port List

| Signal Name | Direction | Width | Description |
| :--- | :--- | :--- | :--- |
| `aclk` | Input | 1 | Global system clock (Nominal: 100 MHz). |
| `aresetn` | Input | 1 | Synchronous active-low reset. |
| **Input Feature Interface** | | | **AXI4-Stream Slave** |
| `s_axis_tdata` | Input | 8 | 8-bit unsigned spectral magnitude ($X_{in}[k]$). |
| `s_axis_tvalid` | Input | 1 | High when upstream FFT engine presents valid bin. |
| `s_axis_tready` | Output | 1 | High when accelerator can accept data. |
| `s_axis_tlast` | Input | 1 | Asserted concurrently with 32nd feature ($k = 31$). |
| **Output Mask Interface** | | | **AXI4-Stream Master** |
| `m_axis_tdata` | Output | 8 | 8-bit unsigned attenuation mask ($M[k] \in [0, 255]$). |
| `m_axis_tvalid` | Output | 1 | Asserted when valid mask byte is available. |
| `m_axis_tready` | Input | 1 | High when downstream multiplier can accept mask. |
| `m_axis_tlast` | Output | 1 | Asserted concurrently with 16th mask value ($k = 15$). |

### 7.2 Execution Timing & Latency Budget

```
          ◄────────── 16 ms Frame Interval (1,600,000 Cycles @ 100 MHz) ──────────►
───┬───────────────────┬─────────────────────┬──────────────────┬─────────────────┬───
   │ FFT & Magnitude   │ Input Streaming     │ MLP Compute Core │ Mask Output     │
   │ (512 cycles)      │ 32 cycles           │ 48 cycles        │ 16 cycles       │
───┴───────────────────┴─────────────────────┴──────────────────┴─────────────────┴───
   ▲                                                            ▲
   │                                                            │
   New Frame Arrives                                            Enhanced Mask Ready
   Latency Margin: > 99.9% Idle / Multi-Channel Capable
```

* **Core Latency:**
  * Ingestion: 32 clock cycles.
  * Layer 1 compute (folded across 4 parallel DSPs): 32 cycles.
  * Layer 2 compute + Sigmoid lookup: 16 cycles.
  * Total Inference Latency: **$< 100$ clock cycles ($< 1.0\text{ }\mu\text{s}$ at 100 MHz)**.
* **Initiation Interval (II):** Target $\text{II} = 1$ (fully pipelined) or $\text{II} = 32$ (resource-shared DSP configuration).
* **Timing Margin:** Processing takes $< 0.01\%$ of the 16 ms hop time, leaving massive slack for low-power dynamic frequency scaling or multi-channel time-division multiplexing.

---

## 8. Verification & Co-Simulation Contract

To ensure 100% bit-exact agreement between Python/PyTorch model simulations and Vivado RTL behavioral testbenches:

1. **Golden Reference Vectors:**
   * `tb_input_features.mem`: 32 lines of hexadecimal input samples.
   * `tb_expected_mask.mem`: 16 lines of expected output mask values.
2. **Acceptance Criterion:**
   $$\max_{k \in [0, 15]} \left| M_{RTL}[k] - M_{Python\_FixedPoint}[k] \right| = 0 \text{ LSB}$$
   Zero bit-drift tolerated between fixed-point Python simulator and RTL.

---
*End of Specification.*
