---
title: Deep Unsupervised Learning using Nonequilibrium Thermodynamics - notes
pubDate: 2025-08-30
categories: ["ML/AI"]
description: "Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli. https://arxiv.org/abs/1503.03585, 2015."
slug: diffusion
---

拡散モデルの原型となる論文. 表題の通り, 非平衡熱力学を下敷きにしており, 情報科学への応用, および定式化, 最終的な実装までをもやってのける強力な知性に驚嘆する. 美しいことにモデルの枠組みも非常に簡明で, 以下 2 つの双対的な過程より構成されている.

1. 拡散過程: 所与の複雑なデータ分布から Gauss 分布や Laplace 分布などの単純な分布への変換, Fig. 1 でいえば, 青 ($t=0$) から 青 ($t=T$) への変換.
2. 逆拡散過程: 単純な分布から所与のデータ分布を再現しようとする変換, すなわち生成過程, Fig. 1 でいえば, 赤 ($t=T$) から 赤 ($t=0$) への変換.
![framework](/blog/20250830_diffusion_framework.png)

## 拡散過程の軌跡

拡散過程は, "データ分布 $q(\mathbf{x}^{(0)})$ が, 単純な分布 $\pi(\mathbf{y})$ に対する Markov 拡散核 $T_{\pi}(\mathbf{y}|\mathbf{y}';β)$ (ここで $β$ は拡散率) を繰り返し適用されることで, 段々と $\pi(\mathbf{y})$ に変換されていく過程" として, 次のように定式化される. ただし, 核のパラメータは多層パーセプトロンにより調整される.
$$
\pi(\mathbf{y}) = ∫T_{\pi}(\mathbf{y}|\mathbf{y}';β)\pi(\mathbf{y}')\mathrm{d}\mathbf{y}'
$$
またその軌跡は, 拡散を $T$ ステップ行うとすれば, 次のような Markov 連鎖として与えられる.
$$
\begin{align}
q\left(\mathbf{x}^{(0\cdots T)}\right) &= q\left(\mathbf{x}^{(0)}\right)∏_{t=1}^Tq\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right) \\
q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right) &= T_{\pi}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)};β_t\right)
\end{align}
$$

## 逆拡散過程の軌跡

拡散過程の定式化を振り返れば, 逆拡散過程の軌跡が $p\left(\mathbf{x}^{(T)}\right)=\pi\left(\mathbf{x}^{(T)}\right)$ として, 次のように与えられることは自明といってよい.
$$
p\left(\mathbf{x}^{(0\cdots T)}\right) = p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^Tp\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)
$$

## 生成分布

逆拡散過程を通じて最終的に得られる生成分布は以下で与えられる.
$$
\begin{align}
&\space\space\space\space p\left(\mathbf{x}^{(0)}\right) \\
&= ∫p\left(\mathbf{x}^{(0\cdots T)}\right)\mathrm{d}\mathbf{x}^{(1\cdots T)}\space\space(*) \\
&= ∫\frac{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}p\left(\mathbf{x}^{(0\cdots T)}\right)\mathrm{d}\mathbf{x}^{(1\cdots T)} \\
&= ∫\frac{p\left(\mathbf{x}^{(0\cdots T)}\right)}{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)\mathrm{d}\mathbf{x}^{(1\cdots T)} \\
&= ∫p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^T\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)\mathrm{d}\mathbf{x}^{(1\cdots T)}\space\space(**) \\
&= \mathbb{E}_{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}\left[p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^T\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right] \\
&\mathop{→}\limits_{β→0} \mathbb{E}_{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}\left[p\left(\mathbf{x}^{(T)}\right)\right]\space\space∵\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}→1
\end{align}
$$
近似のない式変形を行ったが, 各段階でそれぞれの式の表す意味は大きく異なる. $(*)$ においては, $p\left(\mathbf{x}^{(0\cdots T)}\right)$ の規格化定数が計算不能ゆえ, 積分も計算不能であった. しかし $(**)$ において, 焼きなまし重点サンプリング (AIS), ないしは物理的な視点から Jarzynski 等式の発想により, 規格化定数は拡散過程と逆拡散過程の比として現れるようになった. そして最終的に, 拡散率を $0$ とする極限, すなわち統計力学的な見方では準静的過程とよばれる過程を考えることで, その比すらも消失した. ここにおいてはもはや積分計算の際に, Monte Carlo 法による多数サンプリングを必要とせず, $q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)$ から単一サンプリングするだけで結果を導けるようになった.

## 訓練

訓練とは, 次で与えられる対数尤度 $L$ を最大化することである. しかし実際の問題は, その下限 $K$ を最大化する逆拡散過程を見つけることである, $\mathrm{i.e.}$, $\hat{p}\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)≔\mathop{\argmax}\limits_{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}K$.

$$
\begin{align}
&\space\space\space\space L \\
&≔ \mathbb{E}_{q\left(\mathbf{x}^{(0)}\right)}\left[\log p\left(\mathbf{x}^{(0)}\right)\right] \\
&= ∫q\left(\mathbf{x}^{(0)}\right)\log p\left(\mathbf{x}^{(0)}\right)\mathrm{d}\mathbf{x}^{(0)} \\
&= ∫q\left(\mathbf{x}^{(0)}\right)\log\left[\mathbb{E}_{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}\left[p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^T\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\right]\mathrm{d}\mathbf{x}^{(0)} \\
&≥ ∫q\left(\mathbf{x}^{(0)}\right)\mathbb{E}_{q\left(\mathbf{x}^{(1\cdots T)}|\mathbf{x}^{(0)}\right)}\left[\log\left[p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^T\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\right]\mathrm{d}\mathbf{x}^{(0)} \\
&\space\space\space\space∵\text{ Jensen の定理 (準静的過程で等号成立)} \\
&= ∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[p\left(\mathbf{x}^{(T)}\right)∏_{t=1}^T\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&≔ K
\end{align}
$$

### K を解析的に計算可能な形へ書き換える

-- $p\left(\mathbf{x}^{(T)}\right)$ のエントロピー $H_p\left(\mathbf{x}^{(T)}\right)$ を導入 --

$$
\begin{align}
&\space\space\space\space K \\
&= ∫q\left(\mathbf{x}^{(0\cdots T)}\right)\sum_{t=1}^T\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log p\left(\mathbf{x}^{(T)}\right)\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&= \sum_{t=1}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+∫q\left(\mathbf{x}^{(T)}\right)\log p\left(\mathbf{x}^{(T)}\right)\mathrm{d}\mathbf{x}^{(T)} \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+∫q\left(\mathbf{x}^{(0)},\mathbf{x}^{(1)}\right)\log\left[\frac{p\left(\mathbf{x}^{(0)}|\mathbf{x}^{(1)}\right)}{q\left(\mathbf{x}^{(1)}|\mathbf{x}^{(0)}\right)}\right]\mathrm{d}\mathbf{x}^{(0)}\mathrm{d}\mathbf{x}^{(1)} \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
\end{align}
$$

-- $t=0$ におけるエッジ効果を除去 --

$p\left(\mathbf{x}^{(0)}|\mathbf{x}^{(1)}\right)\pi\left(\mathbf{x}^{(1)}\right)=q\left(\mathbf{x}^{(1)}|\mathbf{x}^{(0)}\right)\pi\left(\mathbf{x}^{(0)}\right)$ と考える.
$$
\begin{align}
&\space\space\space\space K \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+∫q\left(\mathbf{x}^{(0)},\mathbf{x}^{(1)}\right)\log\left[\frac{q\left(\mathbf{x}^{(1)}|\mathbf{x}^{(0)}\right)\pi\left(\mathbf{x}^{(0)}\right)}{q\left(\mathbf{x}^{(1)}|\mathbf{x}^{(0)}\right)\pi\left(\mathbf{x}^{(1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0)}\mathrm{d}\mathbf{x}^{(1)} \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+∫q\left(\mathbf{x}^{(0)},\mathbf{x}^{(1)}\right)\log\left[\frac{\pi\left(\mathbf{x}^{(0)}\right)}{\pi\left(\mathbf{x}^{(1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0)}\mathrm{d}\mathbf{x}^{(1)} \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+H_{q,\pi}\left(\mathbf{x}^{(0)}\right)-H_{q,\pi}\left(\mathbf{x}^{(1)}\right) \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
\end{align}
$$
最後の変形は拡散核を適切に設計していれば, $\pi$ に対する $q$ の交差エントロピーが $∀t∈\mathbb{N}, H_{q,\pi}\left(\mathbf{x}^{(t)}\right)=-\mathbb{E}_{q\left(\mathbf{x}^{(t)}\right)}\left[\log\pi\left(\mathbf{x}^{(t)}\right)\right]=\mathrm{Const.}$ と考えられることによる.

-- 事後分布 $q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(0)}\right)$ に着目 --

拡散過程は Markov 過程なので, 条件に $\mathbf{x}^{(0)}$ が入ったところで影響はなく以下が成り立つ.
$$
\begin{align}
∀t≥2,&\space\space\space\space q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right) \\
&= q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)},\mathbf{x}^{(0)}\right) \\
&= \frac{q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)}\right)q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(0)}\right)}{q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(0)}\right)}\space\space∵\text{ Bayes の定理}
\end{align}
$$
したがって $K$ は次のように書き直せる.
$$
\begin{align}
&\space\space\space\space K \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)}\right)}\frac{q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(0)}\right)}{q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(0)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
&= \sum_{t=2}^T∫q\left(\mathbf{x}^{(0\cdots T)}\right)\log\left[\frac{p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)}\right)}\right]\mathrm{d}\mathbf{x}^{(0\cdots T)} \\
&\space\space+\sum_{t=2}^T\left[H_q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(0)}\right)-H_q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(0)}\right)\right] \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
&= -\sum_{t=2}^T∫q\left(\mathbf{x}^{(0)},\mathbf{x}^{(t)}\right)D_{KL}\left(q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)},\mathbf{x}^{(0)}\right)||p\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)\right)\mathrm{d}\mathbf{x}^{(0)}\mathrm{d}\mathbf{x}^{(t)} \\
&\space\space+H_q\left(\mathbf{x}^{(T)}|\mathbf{x}^{(0)}\right)-H_q\left(\mathbf{x}^{(1)}|\mathbf{x}^{(0)}\right) \\
&\space\space-H_p\left(\mathbf{x}^{(T)}\right) \\
\end{align}
$$

## 第二の分布 $r\left(\mathbf{x}^{(t)}\right)$ の導入

十分滑らかな $r\left(\mathbf{x}^{(t)}\right)$ の導入により, 摂動として外部情報を取り込むことで, 理論の軽微な修正のみで事前分布 $p\left(\mathbf{x}^{(0)}\right)$ の再現を超えた柔軟な生成タスクを可能にする. ただし, $r\left(\mathbf{x}^{(t)}\right)$ の導入によっても閉形式となれば, 当然摂動近似によらず厳密計算が可能である.

修正逆拡散分布 $\tilde{p}\left(\mathbf{x}^{(t)}\right)$ を第二の分布を導入して次のようにかく.
$$
\tilde{p}\left(\mathbf{x}^{(t)}\right) = \frac{1}{\tilde{Z}_t}p\left(\mathbf{x}^{(t)}\right)r\left(\mathbf{x}^{(t)}\right)
$$
さらに, 修正 Markov 逆拡散核 $\tilde{p}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)$ を用いれば次のように書き直せる.
$$
\begin{align}
&\tilde{p}\left(\mathbf{x}^{(t)}\right) = ∫\tilde{p}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)\tilde{p}\left(\mathbf{x}^{(t+1)}\right)\mathrm{d}\mathbf{x}^{(t+1)} \\
⇔ &\frac{p\left(\mathbf{x}^{(t)}\right)r\left(\mathbf{x}^{(t)}\right)}{\tilde{Z}_t} = ∫\tilde{p}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)\frac{p\left(\mathbf{x}^{(t+1)}\right)r\left(\mathbf{x}^{(t+1)}\right)}{\tilde{Z}_{t+1}}\mathrm{d}\mathbf{x}^{(t+1)} \\
⇔ &p\left(\mathbf{x}^{(t)}\right) = ∫\tilde{p}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)\frac{\tilde{Z}_tr\left(\mathbf{x}^{(t+1)}\right)}{\tilde{Z}_{t+1}r\left(\mathbf{x}^{(t)}\right)}p\left(\mathbf{x}^{(t+1)}\right)\mathrm{d}\mathbf{x}^{(t+1)}
\end{align}
$$
したがって, 修正逆拡散過程は以下のようにかける.
$$
\begin{align}
\tilde{p}\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right) &= p\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)\frac{\tilde{Z}_{t+1}r\left(\mathbf{x}^{(t)}\right)}{\tilde{Z}_tr\left(\mathbf{x}^{(t+1)}\right)} \\
&≔ \frac{1}{\tilde{Z}_t\left(\mathbf{x}^{(t+1)}\right)}p\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t+1)}\right)r\left(\mathbf{x}^{(t)}\right)
\end{align}
$$

## 逆拡散過程のエントロピー $H_q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)$

$H_q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)$ は解析的に計算可能な上界, および下界によって評価可能である. この評価は逆拡散過程の精度を測るうえで重要である. Bayes の定理により以下の不等式を得る.
$$
H_q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)+H_q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(0)}\right)-H_q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(0)}\right)≤H_q\left(\mathbf{x}^{(t-1)}|\mathbf{x}^{(t)}\right)≤H_q\left(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)}\right)
$$
