import numpy as np
import cv2
from dataclasses import dataclass


def charbonnier(x2, eps=1e-3):
    """ψ(x^2)=sqrt(x^2+eps^2)"""
    return np.sqrt(x2 + eps * eps)


def charbonnier_prime(x2, eps=1e-3):
    """ψ'(x^2)= 1/(2*sqrt(x^2+eps^2))  (对 x^2 的导数)"""
    return 0.5 / np.sqrt(x2 + eps * eps)


def warp_bilinear(img, u, v):
    """
    使用 flow(u,v) 把 img 采样到 (x+u, y+v)
    img: HxW or HxWxC, float32
    u,v: HxW, float32 (像素单位)
    """
    h, w = u.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def grad_xy(img):
    """中心差分梯度，返回 gx, gy"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def laplacian(x):
    """离散拉普拉斯 (4-neighbor)"""
    return (
        -4 * x
        + np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)
    )


@dataclass
class RGBDFlowParams:
    # 论文式 (12) 中的权重：E = Ec + Ez + α Es + γ Eb
    alpha: float = 20.0       # smoothness
    gamma: float = 1.0        # flow magnitude bias
    eps: float = 1e-3         # charbonnier epsilon
    omega: float = 1.7        # SOR parameter
    sor_iters: int = 60       # inner iterations per warp step
    warps: int = 5            # fixed-point outer warps per pyramid level
    pyramid_levels: int = 4
    pyramid_scale: float = 0.5

    # 深度项权重 μ(x)（论文 III-B 说用深度不确定性，近似 d^2）
    # 这里给一个简单模型：sigma(d)=a*d^2+b,  μ= sigma(1)/sigma(d)
    depth_a: float = 1e-3
    depth_b: float = 1e-2

    # w 的像素/米平衡：β=f^2, η=f/z（论文 III-C / (12)）
    focal: float = 525.0      # fx ~ fy
    z_min: float = 0.1        # 避免除 0


class RGBDFlow:
    def __init__(self, params: RGBDFlowParams):
        self.p = params

    def build_pyramid(self, img, levels, scale):
        pyr = [img]
        for _ in range(1, levels):
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            pyr.append(img)
        return pyr

    def depth_mu(self, z):
        # σ(d) ≈ a d^2 + b;  μ = σ(1)/σ(d)
        sigma1 = self.p.depth_a * (1.0 ** 2) + self.p.depth_b
        sigmad = self.p.depth_a * (z ** 2) + self.p.depth_b
        return sigma1 / (sigmad + 1e-12)

    def compute(self, I1, Z1, I2, Z2, init_flow=None):
        """
        I1,I2: HxW 或 HxWx3, float32, [0,1]
        Z1,Z2: HxW, float32, meters, 对齐到 color 坐标
        返回 u,v,w: HxW
        """
        if I1.ndim == 3:
            # 灰度更贴近推导；你也可以改成对 RGB 分通道求和
            I1g = cv2.cvtColor((I1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            I2g = cv2.cvtColor((I2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            I1g, I2g = I1.astype(np.float32), I2.astype(np.float32)

        I1p = self.build_pyramid(I1g, self.p.pyramid_levels, self.p.pyramid_scale)
        I2p = self.build_pyramid(I2g, self.p.pyramid_levels, self.p.pyramid_scale)
        Z1p = self.build_pyramid(Z1.astype(np.float32), self.p.pyramid_levels, self.p.pyramid_scale)
        Z2p = self.build_pyramid(Z2.astype(np.float32), self.p.pyramid_levels, self.p.pyramid_scale)

        # 初始化 flow
        h0, w0 = I1p[-1].shape
        if init_flow is None:
            u = np.zeros((h0, w0), np.float32)
            v = np.zeros((h0, w0), np.float32)
            w = np.zeros((h0, w0), np.float32)
        else:
            u, v, w = init_flow
            # 缩放到最粗层
            scale_to_coarse = (self.p.pyramid_scale ** (self.p.pyramid_levels - 1))
            u = cv2.resize(u, (w0, h0), interpolation=cv2.INTER_LINEAR) * scale_to_coarse
            v = cv2.resize(v, (w0, h0), interpolation=cv2.INTER_LINEAR) * scale_to_coarse
            w = cv2.resize(w, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # coarse-to-fine
        for lvl in reversed(range(self.p.pyramid_levels)):
            I1l, I2l = I1p[lvl], I2p[lvl]
            Z1l, Z2l = Z1p[lvl], Z2p[lvl]
            h, wid = I1l.shape

            if u.shape != (h, wid):
                # 上采样到当前层（u,v 需要按尺度放大）
                scale_up = 1.0 / self.p.pyramid_scale
                u = cv2.resize(u, (wid, h), interpolation=cv2.INTER_LINEAR) * scale_up
                v = cv2.resize(v, (wid, h), interpolation=cv2.INTER_LINEAR) * scale_up
                w = cv2.resize(w, (wid, h), interpolation=cv2.INTER_LINEAR)

            # 预计算 I1, Z1 的梯度（论文里线性化使用 I1 的梯度项）
            I1x, I1y = grad_xy(I1l)
            Z1x, Z1y = grad_xy(Z1l)

            # 深度权重 μ(x)
            mu = self.depth_mu(np.maximum(Z1l, self.p.z_min)).astype(np.float32)

            # 固定点 outer loop：warp -> 线性化 -> SOR
            for _ in range(self.p.warps):
                # warp I2, Z2 到 I1 坐标
                I2w = warp_bilinear(I2l, u, v)
                Z2w = warp_bilinear(Z2l, u, v)

                # 残差 ΔtI, ΔtZ（论文式 (2)(7)）
                dtI = (I2w - I1l).astype(np.float32)
                dtZ = (Z2w - Z1l).astype(np.float32)

                # 鲁棒权重 ψ'(·)
                psiD = charbonnier_prime(dtI * dtI, eps=self.p.eps).astype(np.float32)
                # 深度项对 (dtZ - w)^2 做 ψ'
                rz = (dtZ - w).astype(np.float32)
                psiZ = charbonnier_prime(rz * rz, eps=self.p.eps).astype(np.float32)

                # 平滑项权重 ψ'(|∇u|^2+|∇v|^2+β|∇w|^2)，这里用拉普拉斯近似，不显式算 ∇
                beta = (self.p.focal ** 2)
                # 用当前 u,v,w 的“roughness”近似
                ru = laplacian(u)
                rv = laplacian(v)
                rw = laplacian(w)
                smooth_measure = (ru * ru + rv * rv + beta * (rw * rw)).astype(np.float32)
                psiS = charbonnier_prime(smooth_measure, eps=self.p.eps).astype(np.float32)

                # 内层：SOR 解 du,dv,dw（论文 III-D 思路）
                du = np.zeros_like(u)
                dv = np.zeros_like(v)
                dw = np.zeros_like(w)

                # η(x)=f/z（论文 (12)）
                eta = (self.p.focal / np.maximum(Z1l, self.p.z_min)).astype(np.float32)

                # 为了速度，先把常用量缓存
                I1x2 = I1x * I1x
                I1y2 = I1y * I1y
                I1xy = I1x * I1y

                Z1x2 = Z1x * Z1x
                Z1y2 = Z1y * Z1y
                Z1xy = Z1x * Z1y

                a = self.p.alpha
                g = self.p.gamma
                wSOR = self.p.omega

                for _it in range(self.p.sor_iters):
                    # 拉普拉斯项（用 ψS 做权重，简单乘进去）
                    Lu = laplacian(u + du)
                    Lv = laplacian(v + dv)
                    Lw = laplacian(w + dw)

                    # ---- 更新 du ----
                    # 线性化：dtI + I1x*du + I1y*dv
                    #        dtZ + Z1x*du + Z1y*dv - dw
                    # 分母近似：psiD*I1x^2 + mu*psiZ*Z1x^2 + g + a*(something)
                    # 这里把平滑项用 a*psiS*Lu 形式并入 num
                    num_u = (
                        - psiD * I1x * (dtI + I1y * dv)
                        - mu * psiZ * Z1x * (dtZ + Z1y * dv - dw)
                        + a * psiS * Lu
                        - g * u
                    )
                    denom_u = (psiD * I1x2 + mu * psiZ * Z1x2 + g + a * psiS * 4.0)  # 4 近似拉普拉斯对中心系数
                    du = (1 - wSOR) * du + wSOR * (num_u / (denom_u + 1e-12))

                    # ---- 更新 dv ----
                    num_v = (
                        - psiD * I1y * (dtI + I1x * du)
                        - mu * psiZ * Z1y * (dtZ + Z1x * du - dw)
                        + a * psiS * Lv
                        - g * v
                    )
                    denom_v = (psiD * I1y2 + mu * psiZ * Z1y2 + g + a * psiS * 4.0)
                    dv = (1 - wSOR) * dv + wSOR * (num_v / (denom_v + 1e-12))

                    # ---- 更新 dw ----
                    # 深度残差项对 dw： (dtZ + Z1x*du + Z1y*dv - dw)
                    num_w = (
                        mu * psiZ * (dtZ + Z1x * du + Z1y * dv)
                        + a * psiS * beta * Lw
                        - g * eta * w
                    )
                    denom_w = (mu * psiZ + g * eta + a * psiS * beta * 4.0)
                    dw = (1 - wSOR) * dw + wSOR * (num_w / (denom_w + 1e-12))

                # 累加增量
                u += du
                v += dv
                w += dw

        return u, v, w


if __name__ == "__main__":
    # 示例：用两帧 RGB + Depth（需要你自己替换路径/读入方式）
    # I: BGR [0,1], Z: meters float32
    I1 = cv2.imread("color_000.png")
    I2 = cv2.imread("color_001.png")
    Z1 = cv2.imread("depth_000.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    Z2 = cv2.imread("depth_001.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    I1 = I1.astype(np.float32) / 255.0
    I2 = I2.astype(np.float32) / 255.0

    params = RGBDFlowParams(
        alpha=20.0,
        gamma=0.5,
        sor_iters=80,
        warps=5,
        pyramid_levels=4,
        focal=525.0
    )
    solver = RGBDFlow(params)
    u, v, w = solver.compute(I1, Z1, I2, Z2)

    # 可视化 xy-flow（HSV）
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), np.float32)
    hsv[..., 0] = ang / 2.0
    hsv[..., 1] = 1.0
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    flow_bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite("xy_flow_vis.png", flow_bgr)

    # z-flow 可视化（灰度）
    wz = cv2.normalize(w, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("z_flow_vis.png", wz.astype(np.uint8))