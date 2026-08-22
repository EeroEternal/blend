/// 张量数据类型。量化格式对齐 FreeToken 支持集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    /// NVFP4（nvidia 4-bit 浮点，block scale）
    NvFp4,
    MxFp4,
    /// DeepSeek-V4 专用 fp4 格式
    DsFp4,
    Q4_0,
    I32,
}

impl Dtype {
    pub fn bits(self) -> u32 {
        match self {
            Dtype::F32 | Dtype::I32 => 32,
            Dtype::F16 | Dtype::Bf16 | Dtype::Fp8E4m3 => 16,
            Dtype::NvFp4 | Dtype::MxFp4 | Dtype::DsFp4 | Dtype::Q4_0 => 4,
        }
    }
    /// 专家权重字节数（含 block scale 开销的近似系数）
    pub fn bytes_per_param(self) -> f64 {
        match self {
            Dtype::NvFp4 | Dtype::MxFp4 | Dtype::DsFp4 => 4.5 / 8.0 + 0.0625, // 经验值
            _ => self.bits() as f64 / 8.0,
        }
    }
}
