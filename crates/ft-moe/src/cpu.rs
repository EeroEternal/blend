//! CPU MoE 执行器。
//!
//! P3 目标：AVX512-BF16/VNNI 内核（对齐 FreeToken csrc/cpu_moe，实测 155GB/s）。
//! 当前先提供 f32 参考实现——数值正确性是后续 SIMD 优化的 golden 基线。

use ft_core::FtError;

/// 单层 MoE 的 CPU 执行契约。
///
/// 形状约定（与 FreeToken 一致）：
/// * h: [num_tokens, hidden]
/// * w13: [num_experts, 2*inter, hidden]  (gate+up 合并)
/// * w2:  [num_experts, hidden, inter]    (down)
/// * topk: [num_tokens, k] 专家索引
/// * weights: [num_tokens, k] 路由权重
pub trait CpuMoeExecutor: Send + Sync {
    fn forward(
        &self,
        h: &mut [f32],
        num_tokens: usize,
        hidden: usize,
        w13: &[f32],
        w2: &[f32],
        inter: usize,
        num_experts: usize,
        topk: &[u32],
        weights: &[f32],
        k: usize,
    ) -> Result<(), FtError>;
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 参考实现：朴素 f32 循环。只求对拍正确，不求速度。
pub struct NaiveF32Executor;

impl CpuMoeExecutor for NaiveF32Executor {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &mut [f32],
        num_tokens: usize,
        hidden: usize,
        w13: &[f32],
        w2: &[f32],
        inter: usize,
        num_experts: usize,
        topk: &[u32],
        weights: &[f32],
        k: usize,
    ) -> Result<(), FtError> {
        if h.len() != num_tokens * hidden {
            return Err(FtError::Invalid("h size mismatch".into()));
        }
        if topk.len() != num_tokens * k || weights.len() != num_tokens * k {
            return Err(FtError::Invalid("topk/weights size mismatch".into()));
        }
        let mut acc = vec![0f32; hidden];
        let mut gate_up = vec![0f32; 2 * inter];
        let mut mid = vec![0f32; inter];

        for t in 0..num_tokens {
            acc.fill(0.0);
            for e in 0..k {
                let expert = topk[t * k + e] as usize;
                if expert >= num_experts {
                    return Err(FtError::Invalid(format!("expert id {expert} >= {num_experts}")));
                }
                let rw = weights[t * k + e];
                if rw == 0.0 {
                    continue;
                }
                // gate+up
                for i in 0..(2 * inter) {
                    let row = &w13[expert * 2 * inter * hidden + i * hidden..][..hidden];
                    let x = &h[t * hidden..][..hidden];
                    let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                    gate_up[i] = dot;
                }
                // silu(gate) * up -> mid
                for i in 0..inter {
                    mid[i] = silu(gate_up[i]) * gate_up[inter + i];
                }
                // down
                for o in 0..hidden {
                    let row = &w2[expert * hidden * inter + o * inter..][..inter];
                    let dot: f32 = row.iter().zip(mid.iter()).map(|(a, b)| a * b).sum();
                    acc[o] += rw * dot;
                }
            }
            h[t * hidden..][..hidden].copy_from_slice(&acc);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 微型 MoE：1 token, hidden=2, inter=1, 2 专家, topk=1。
    /// 手算数值作为 golden。
    #[test]
    fn naive_executor_matches_hand_computed() {
        let ex = NaiveF32Executor;
        let mut h = vec![1.0, 2.0]; // 输入
        // 专家 0：w13 = [[1,0],[0,1], [1,1],[0,0]] (gate=[1,0],up=[0,1],再补一组凑 2*inter)
        // inter=1 → w13 形状 [2, 2, 2]，w2 形状 [2, 2, 1]
        let w13 = vec![
            1.0, 0.0, // expert0 gate row
            0.0, 1.0, // expert0 up row
            1.0, 1.0, // expert1 gate row
            0.0, 0.0, // expert1 up row
        ];
        let w2 = vec![2.0, 3.0, 1.0, 1.0]; // expert0 down=[2,3], expert1 down=[1,1]
        let topk = vec![0u32];
        let weights = vec![0.5];

        ex.forward(&mut h, 1, 2, &w13, &w2, 1, 2, &topk, &weights, 1).unwrap();

        // gate = [1*1+0*2] = 1; up = [0*1+1*2] = 2
        // mid = silu(1)*2 = (1/(1+e^-1))*2 ≈ 1.462117
        // out = 0.5 * [mid*2, mid*3] = [1.462117, 2.193176]
        assert!((h[0] - 1.462117).abs() < 1e-5, "got {}", h[0]);
        assert!((h[1] - 2.193176).abs() < 1e-5, "got {}", h[1]);
    }

    #[test]
    fn rejects_bad_shapes_and_ids() {
        let ex = NaiveF32Executor;
        // h 长度错
        let mut h2 = vec![0.0; 3];
        assert!(matches!(
            ex.forward(&mut h2, 1, 2, &[0.0; 8], &[0.0; 4], 1, 2, &[0], &[0.5], 1),
            Err(FtError::Invalid(_))
        ));
        // topk 长度错
        let mut h3 = vec![0.0; 2];
        assert!(matches!(
            ex.forward(&mut h3, 1, 2, &[0.0; 8], &[0.0; 4], 1, 2, &[], &[], 1),
            Err(FtError::Invalid(_))
        ));
        // 专家 id 越界
        let mut h4 = vec![0.0; 2];
        assert!(matches!(
            ex.forward(&mut h4, 1, 2, &[0.0; 8], &[0.0; 4], 1, 2, &[5], &[0.5], 1),
            Err(FtError::Invalid(_))
        ));
    }
}
