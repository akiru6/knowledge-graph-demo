# Knowledge Graph Root Cause Analysis Demo (Neo4j + Groq + Cohere)

A practical demonstration of how to use a Knowledge Graph for root cause analysis by combining structured data (CSVs) and unstructured data (text) using the `neo4j-graphrag` library, powered by Groq for fast LLM inference and Cohere for embeddings.

[简体中文](#简体中文)

---

### 🎯 Project Goal

This project simulates a real-world business scenario for a fictional furniture company, "Cheese妙想" . When a customer complaint about a "wobbly table" is received, we use a Knowledge Graph to automatically trace the issue from the product, through its specific parts, down to the exact supplier responsible for the faulty component.

This demonstrates the power of "RAG on Graphs"—unifying disparate data sources to uncover hidden relationships and derive actionable insights.

### ⚙️ How It Works

The entire process is orchestrated by the `issue_demo.py` script, which performs the following steps:

1.  **Build Domain Graph**: It reads structured data from CSV files (`products.csv`, `parts.csv`, `suppliers.csv`) to create a foundational Knowledge Graph representing the company's supply chain. This is our "ground truth".
2.  **Build Subject Graph**: It uses an LLM (Groq) to read and understand an unstructured customer complaint (`complaint_report.md`). It extracts key entities (like the product name and the issue) and their relationships, creating a separate, small graph.
3.  **Bridge the Graphs**: It intelligently links the Subject Graph to the Domain Graph by matching entity names (e.g., connecting the "wobbly table" from the complaint to the official "wobbly table" product in our database). This is the crucial entity resolution step.
4.  **Perform Root Cause Analysis**: It runs a single Cypher query across the unified graph to find the complete path from the "wobble issue" to the "supplier", instantly identifying the root cause.


### 🚀 Getting Started

#### Prerequisites

*   Python 3.9+
*   An active [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) instance (free tier is sufficient).
*   A [Groq API Key](https://console.groq.com/keys).
*   A [Cohere API Key](https://dashboard.cohere.com/api-keys).

#### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    *   Rename the `.env.sample` file to `.env`.
    *   Open the `.env` file and fill in your credentials for Neo4j, Groq, and Cohere.

5.  **Run the demo:**
    ```bash
    python issue_demo.py
    ```

### 📄 File Structure

*   `issue_demo.py`: The main script that orchestrates the entire workflow.
*   `CustomGroqLLM.py`: A custom adapter to make the Groq LLM compatible with the `neo4j-graphrag` library.
*   `requirements.txt`: A list of all necessary Python packages.
*   `.env.sample`: A template for the environment variables file. You need to rename this to `.env` and add your keys.
*   `.gitignore`: Specifies which files (like `.env`) should not be tracked by Git.

---

## 简体中文

### 🎯 项目目标

本项目模拟了一个真实的商业场景：一家名为“Cheese妙想”的创意家居公司，在收到关于“桌子晃动”的客户投诉后，如何利用知识图谱自动进行根因分析，从具体的产品追溯到其问题零件，并最终定位到提供该零件的供应商。

这个 Demo 展示了 "RAG on Graphs" (基于图谱的检索增强生成) 的强大能力——即融合异构数据源，以发现隐藏的关联，并获得可执行的洞察。

### ⚙️ 技术实现流程

整个流程由 `issue_demo.py` 脚本全自动执行，包含以下四个核心步骤：

1.  **构建领域图谱 (Domain Graph)**: 脚本首先读取 CSV 文件中的结构化数据（产品、零件、供应商信息），构建出代表公司供应链的“核心事实”知识图谱。
2.  **构建主题图谱 (Subject Graph)**: 接着，脚本利用大语言模型 (Groq) 读取并理解非结构化的客户投诉邮件 (.md 文件)，从中提取出关键实体（如产品名、问题描述）及其关系，形成一个独立的“事件”小图谱。
3.  **连接图谱 (The Bridge)**: 通过匹配实体名称（例如，将投诉邮件中的“哥德堡摇摇桌”与数据库中的官方产品节点关联），脚本巧妙地在领域图谱和主题图谱之间建立起一座桥梁。这是实现数据打通的“实体解析”关键步骤。
4.  **执行根因分析**: 最后，脚本在融合后的全图上执行一次 Cypher 查询，便能找到从“晃动问题”到“供应商”的完整路径， мгновенно 锁定问题的根源。


### 🚀 快速开始

#### 环境要求

*   Python 3.9+
*   一个可用的 [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) 数据库实例 (免费版即可)。
*   一个 [Groq API 密钥](https://console.groq.com/keys)。
*   一个 [Cohere API 密钥](https://dashboard.cohere.com/api-keys)。

#### 安装与配置

1.  **克隆代码库:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows 用户请运行 `venv\Scripts\activate`
    ```

3.  **安装依赖包:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量:**
    *   将 `.env.sample` 文件重命名为 `.env`。
    *   打开 `.env` 文件，填入你自己的 Neo4j, Groq, 和 Cohere 的访问凭证。

5.  **运行 Demo:**
    ```bash
    python issue_demo.py
    ```

### 📄 文件结构

*   `issue_demo.py`: 驱动整个流程的核心脚本。
*   `CustomGroqLLM.py`: 一个自定义的适配器，用于让 Groq LLM 能够兼容 `neo4j-graphrag` 库。
*   `requirements.txt`: 项目所需的所有 Python 依赖包列表。
*   `.env.sample`: 环境变量的模板文件，你需要将它重命名为 `.env` 并填入你的密钥。
*   `.gitignore`: 用于指定哪些文件（如 `.env`）不应被 Git追踪，以防密钥泄露。