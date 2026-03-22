import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const sections = [
  {
    title: 'LLM 推理',
    desc: '从 tokenize 到 sampling 的完整推理链路，以及 vLLM、PagedAttention、投机解码等优化手段',
    link: '/docs/category/llm推理',
    count: 3,
  },
  {
    title: 'LLM 训练',
    desc: '训练全流程、3D并行、ZeRO、梯度累积、混合精度、RLHF 与 DPO',
    link: '/docs/category/llm训练',
    count: 6,
  },
  {
    title: '数学推导',
    desc: 'Attention、RoPE、Softmax、反向传播、Adam — 手推一遍才是真的会',
    link: '/docs/category/数学推导',
    count: 1,
  },
  {
    title: '编程题',
    desc: '手撕 MHA / GQA、写 CUDA kernel、实现 PagedAttention 的 block allocator',
    link: '/docs/category/编程题',
    count: 1,
  },
  {
    title: '论文解读',
    desc: 'Transformer、PagedAttention、DeepSeek-V3 — 读懂原始论文，面试才有底气',
    link: '/docs/category/论文解读',
    count: 3,
  },
  {
    title: '面经',
    desc: '字节、美团、快手、NVIDIA、百度 — 真实面试题与复盘',
    link: '/docs/category/面经',
    count: 4,
  },
  {
    title: '进阶专题',
    desc: '多模态、Agent、Scaling Law、线上问题排查',
    link: '/docs/category/进阶专题',
    count: 4,
  },
  {
    title: '速查手册',
    desc: '66 道高频题的 30 秒口述答案，面试前最后过一遍',
    link: '/docs/category/速查手册',
    count: 3,
  },
];

export default function Home() {
  return (
    <Layout title="首页" description="LLM & AI Infra 知识库">
      <div className={styles.page}>
        <header className={styles.hero}>
          <p className={styles.greeting}>Hi, 这里是</p>
          <h1 className={styles.name}>Zhiyuan 的 LLM 笔记</h1>
          <p className={styles.intro}>
            整理了 LLM 推理 / 训练 / 系统优化方向的面试知识点，
            <br />
            从原理推导到代码实现，从论文解读到真实面经。
          </p>
          <div className={styles.heroLinks}>
            <Link to="/docs/inference/pipeline" className={styles.primaryBtn}>
              从推理 Pipeline 开始
            </Link>
            <Link to="/docs/category/速查手册" className={styles.secondaryBtn}>
              面试速查
            </Link>
          </div>
        </header>

        <section className={styles.grid}>
          <h2 className={styles.sectionTitle}>目录</h2>
          <div className={styles.cards}>
            {sections.map((s) => (
              <Link key={s.title} to={s.link} className={styles.card}>
                <div className={styles.cardHead}>
                  <span className={styles.cardTitle}>{s.title}</span>
                  <span className={styles.badge}>{s.count} 篇</span>
                </div>
                <p className={styles.cardDesc}>{s.desc}</p>
              </Link>
            ))}
          </div>
        </section>

        <section className={styles.recentSection}>
          <h2 className={styles.sectionTitle}>推荐阅读顺序</h2>
          <ol className={styles.readingOrder}>
            <li>
              <Link to="/docs/inference/pipeline">推理 Pipeline 面试题</Link>
              <span className={styles.tag}>必读</span>
              — 先建立完整的推理链路认知
            </li>
            <li>
              <Link to="/docs/training/pipeline">训练 Pipeline 面试题</Link>
              <span className={styles.tag}>必读</span>
              — 理解从 data loading 到 checkpoint 的全流程
            </li>
            <li>
              <Link to="/docs/math/derivations">数学推导</Link>
              — 手推 Attention、RoPE、反向传播等核心公式
            </li>
            <li>
              <Link to="/docs/papers/attention-is-all-you-need">论文：Attention Is All You Need</Link>
              — 回到原点，重读 Transformer
            </li>
            <li>
              <Link to="/docs/coding/problems">编程题</Link>
              — 手撕代码，检验理解深度
            </li>
            <li>
              <Link to="/docs/quick-ref/quick-answers">30 秒速答</Link>
              — 面试前最后过一遍高频题
            </li>
          </ol>
        </section>
      </div>
    </Layout>
  );
}
