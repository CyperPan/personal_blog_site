import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

const categories = [
  {
    title: 'LLM推理',
    emoji: '🚀',
    description: '推理Pipeline、PagedAttention、投机解码、vLLM、连续批处理、量化优化',
    link: '/docs/category/llm推理',
    color: '#4facfe',
  },
  {
    title: 'LLM训练',
    emoji: '🏋️',
    description: '训练Pipeline、分布式训练、显存优化、通信优化、RLHF/DPO、LoRA',
    link: '/docs/category/llm训练',
    color: '#43e97b',
  },
  {
    title: '数学推导',
    emoji: '📐',
    description: 'Attention、RoPE、Softmax、LayerNorm、反向传播、Adam优化器、MoE路由',
    link: '/docs/category/数学推导',
    color: '#fa709a',
  },
  {
    title: '编程题',
    emoji: '💻',
    description: '手撕MHA/GQA、CUDA Kernel、C++并发、PagedAttention Block分配器',
    link: '/docs/category/编程题',
    color: '#a18cd1',
  },
  {
    title: '面经',
    emoji: '🎯',
    description: '字节、美团、快手、NVIDIA、百度等公司AI Infra真实面试题',
    link: '/docs/category/面经',
    color: '#fbc2eb',
  },
  {
    title: '进阶专题',
    emoji: '🔬',
    description: '多模态训练、Agent架构、Scaling Law、问题排查方法论',
    link: '/docs/category/进阶专题',
    color: '#f6d365',
  },
  {
    title: '速查手册',
    emoji: '📖',
    description: '30秒速答66题、面试技巧模板、学习资源推荐',
    link: '/docs/category/速查手册',
    color: '#89f7fe',
  },
];

function HeroBanner() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/category/llm推理">
            开始学习 →
          </Link>
        </div>
      </div>
    </header>
  );
}

function CategoryCard({title, emoji, description, link, color}) {
  return (
    <div className={clsx('col col--4', styles.cardCol)}>
      <Link to={link} className={styles.cardLink}>
        <div className={styles.card} style={{'--card-color': color}}>
          <div className={styles.cardEmoji}>{emoji}</div>
          <h3 className={styles.cardTitle}>{title}</h3>
          <p className={styles.cardDescription}>{description}</p>
        </div>
      </Link>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="首页" description={siteConfig.tagline}>
      <HeroBanner />
      <main className={styles.main}>
        <div className="container">
          <div className="row">
            {categories.map((props, idx) => (
              <CategoryCard key={idx} {...props} />
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}
