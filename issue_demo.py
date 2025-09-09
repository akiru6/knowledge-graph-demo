# 客户投诉demo.py

import os
import csv
import sys
import re
import asyncio
from pathlib import Path
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path
from dotenv import load_dotenv

from CustomGroqLLM import CustomGroqLLM
from neo4j_graphrag.embeddings import CohereEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo

# --- 【1】加载项目配置 ---

print("--- 准备工作: 正在加载配置和 API 密钥... ---")
load_dotenv()

# Neo4j Aura 数据库连接信息
AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")
AURA_AUTH = (AURA_USER, AURA_PASSWORD)

# Groq - 语言模型
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b" 

# Cohere - 嵌入模型
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBEDDING_MODEL = "embed-english-v3.0"

if not all([AURA_URI, AURA_PASSWORD, GROQ_API_KEY, COHERE_API_KEY]):
    print("\n[出错了] 检查一下你的 .env 文件，好像缺少了必要的 API 密钥。")
    sys.exit()


print("配置加载完成！")
# ----------------------------------------------------

# --- 【2】准备演示需要用到的数据 ---
def prepare_data_and_ai_schema():

    print("--- 准备工作: 正在创建演示用的数据和 AI 的分析蓝图... ---")
    os.makedirs("data", exist_ok=True)

    # 创建 CSV 文件 (我们的产品、零件和供应商信息)
    products_csv = "data/products.csv"
    with open(products_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "product_name"])
        writer.writerow(["P001", "哥德堡摇摇桌"])
        writer.writerow(["P002", "斯德哥尔摩稳稳椅"])
    
    parts_csv = "data/parts.csv"
    with open(parts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_id", "part_name", "product_id"])
        writer.writerow(["L01", "不靠谱桌腿", "P001"])
        writer.writerow(["D01", "结实桌面", "P001"])
        writer.writerow(["C01", "坚固椅背", "P002"])

    suppliers_csv = "data/suppliers.csv"
    with open(suppliers_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["supplier_id", "supplier_name"])
        writer.writerow(["S-A", "便宜木材厂"])
        writer.writerow(["S-B", "良心五金厂"])

    part_supplier_mapping_csv = "data/part_supplier_mapping.csv"
    with open(part_supplier_mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_id", "supplier_id"])
        writer.writerow(["L01", "S-A"])
        writer.writerow(["D01", "S-A"])
        writer.writerow(["C01", "S-B"])
    
    print("产品、零件和供应商的 CSV 文件已经创建好了。")

    # 创建 Markdown 文件 (模拟的客户投诉邮件)
    md_file_path = "data/complaint_report.md"
    md_data = """
    # 紧急客诉：哥德堡摇摇桌质量问题
    
    客户王先生投诉，他购买的“哥德堡摇摇桌”出现了严重的晃动问题。
    经过初步沟通，问题似乎出在桌腿上，感觉非常不稳固。
    王先生非常生气，要求我们立刻解决，否则就要在社交媒体上曝光我们“晃晃家居”！
    """
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(md_data)
    print(f"客户的投诉报告 '{md_file_path}' 也准备好了。")

    # 定义 AI 的分析蓝图 (Schema)
    node_types = [{"label": "Product", "properties": [{"name": "name", "type": "STRING"}]},
                  {"label": "Issue", "properties": [{"name": "name", "type": "STRING"}]}]
    patterns = [['Product', 'HAS_ISSUE', 'Issue']]
    relationship_types = [pattern[1] for pattern in patterns]

    entity_schema = {"node_types": node_types, "relationship_types": relationship_types,
                     "patterns": patterns, "additional_node_types": False}
    print("AI 的分析蓝图 (Schema) 也定义好了，告诉它要关注哪些信息。")
    return (products_csv, parts_csv, suppliers_csv, part_supplier_mapping_csv, md_file_path, entity_schema)


# --- 【3】辅助函数和类 (这部分保持不变) ---
class RegexTextSplitter(TextSplitter):
    def __init__(self, re_str: str): self.re = re_str
    async def run(self, text: str) -> TextChunks:
        texts = re.split(self.re, text); return TextChunks(chunks=[TextChunk(text=str(t), index=i) for i, t in enumerate(texts)])

class MarkdownDataLoader(DataLoader):
    def extract_title(self, text):
        match = re.search(r'^# (.+)$', text, re.MULTILINE); return match.group(1) if match else "无标题报告"
    async def run(self, filepath: Path, metadata={}) -> PdfDocument:
        with open(filepath, "r", encoding="utf-8") as f: text = f.read()
        return PdfDocument(text=text, document_info=DocumentInfo(path=str(filepath), metadata={"title": self.extract_title(text)}))

def run_query(driver, query, params={}):
    with driver.session() as session: return [record.data() for record in session.run(query, params)]

def run_path_query(driver, query, params={}):
    with driver.session() as session: return [record for record in session.run(query, params)]

def clear_database(driver):

    print("\n--- 准备工作: 清空数据库，确保一个干净的开始... ---")
    run_query(driver, "MATCH (n) DETACH DELETE n")
    try:
        constraints = run_query(driver, "SHOW CONSTRAINTS YIELD name")
        for c in constraints: run_query(driver, f"DROP CONSTRAINT `{c['name']}`")
    except Exception: pass
    print("数据库已清空！")
    
def build_supply_chain_graph(driver, products_csv, parts_csv, suppliers_csv, part_supplier_mapping_csv):
    print("\n--- 第 1 步: 导入 CSV 数据，构建我们的供应链网络... ---")
    
    run_query(driver, "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
    run_query(driver, "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.id IS UNIQUE")
    run_query(driver, "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE")

    print("  - 正在导入产品信息...")
    with open(products_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            run_query(driver, "MERGE (p:Product {id: $product_id}) SET p.name = $product_name", row)

    print("  - 正在导入零件，并和产品关联起来...")
    with open(parts_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            run_query(driver, """
                MATCH (p:Product {id: $product_id})
                MERGE (part:Part {id: $part_id}) SET part.name = $part_name
                MERGE (p)-[:HAS_PART]->(part)
                """, row)

    print("  - 正在导入供应商信息...")
    with open(suppliers_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            run_query(driver, "MERGE (s:Supplier {id: $supplier_id}) SET s.name = $supplier_name", row)
    
    print("  - 正在建立零件和供应商的供货关系...")
    with open(part_supplier_mapping_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            run_query(driver, """
                MATCH (part:Part {id: $part_id})
                MATCH (s:Supplier {id: $supplier_id})
                MERGE (part)-[:SUPPLIED_BY]->(s)
                """, row)
    print("供应链网络构建完成！")

class CustomCohereEmbeddings(CohereEmbeddings):
    def embed_query(self, text: str, **kwargs) -> list[float]:
        return super().embed_query(text, input_type="search_document", **kwargs)
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return super().embed_documents(texts, input_type="search_document", **kwargs)

async def extract_complaint_graph_from_text(driver, md_path, entity_schema):

    print("\n--- 第 2 步: 让 AI 读取投诉报告，自动提取关键信息... ---")
    
    print(f"  - 使用 Groq (模型: {GROQ_MODEL}) 来快速理解文本内容。")
    llm = CustomGroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    
    print(f"  - 使用 Cohere (模型: {COHERE_EMBEDDING_MODEL}) 来精准捕捉语义。")
    embedder = CustomCohereEmbeddings(api_key=COHERE_API_KEY, model=COHERE_EMBEDDING_MODEL)
    
    kg_builder = SimpleKGPipeline(llm=llm, driver=driver, embedder=embedder, from_pdf=True,
                                  pdf_loader=MarkdownDataLoader(), text_splitter=RegexTextSplitter("---"),
                                  schema=entity_schema)
    
    await kg_builder.run_async(file_path=str(md_path))

    print("  - AI 分析完成！投诉报告里的关键信息已经被提取并存入图中。")

def build_the_bridge(driver):
    print("\n--- 第 3 步: 关联数据，把投诉信息和我们的产品数据连接起来... ---")
    result = run_query(driver, """
        MATCH (subject:Product:__Entity__), (domain:Product)
        WHERE NOT domain:__Entity__ AND subject.name = domain.name
        MERGE (subject)-[r:CORRESPONDS_TO]->(domain)
        RETURN count(r) as bridges_built
    """)
    if result and result[0]['bridges_built'] > 0:

        print(f"  - 成功连接了 {result[0]['bridges_built']} 个节点！现在投诉和产品数据打通了。")
    else:
        print("  - 警告: 没有找到可以关联的产品。")
    print("数据融合完成！")

def find_the_culprit(driver):

    print("\n--- 第 4 步: 开始溯源！从投诉找到问题的根源... ---")
    print("  - 正在查询知识图谱，寻找从‘晃动问题’到‘供应商’的完整路径...")
    
    final_query = """
    MATCH 
      p1 = (issue:Issue)<-[:HAS_ISSUE]-(p_subject:Product)-[:CORRESPONDS_TO]->(p_domain:Product {name: "哥德堡摇摇桌"}),
      p2 = (p_domain)-[:HAS_PART]->(part:Part)-[:SUPPLIED_BY]->(supplier:Supplier)
    WHERE part.name CONTAINS '腿' AND issue.name CONTAINS '晃动'
    RETURN p1, p2
    """
    
    results = run_path_query(driver, final_query)
    
    if not results:
        print("\n>>> 没有找到完整的溯源路径。可以检查一下数据是否都正确导入了。")
        return
        
    print("\n\n>>> 找到啦！这就是问题的完整溯源路径： <<<")
    
    # 这里直接输出结论，更像Vlog的风格
    print("\n客户投诉的【哥德堡摇摇桌】，问题出在零件【不靠谱桌腿】上。")
    print("而这个零件的供应商是 ---> 【便宜木材厂】！")
    print("\n这样一来，我们就能非常精准地联系供应商去解决问题了。")

    print("\n想在 Neo4j 里亲眼看看这条关系链吗？")
    print("把下面的查询语句复制到 Neo4j Browser 里运行，就能看到可视化的路径了：")
    print("--------------------------------------------------")
    print(final_query.strip())
    print("--------------------------------------------------")

# --- 【5】主函数 ---
async def main():
    driver = None
    try:

        driver = GraphDatabase.driver(AURA_URI, auth=AURA_AUTH)
        driver.verify_connectivity()
        print("\n成功连接到 Neo4j Aura 数据库！")
    except Exception as e:
        print(f"\n[错误] 数据库连接失败: {e}")
        return

    files_and_schema = prepare_data_and_ai_schema()
    products_csv, parts_csv, suppliers_csv, mapping_csv, md_path, schema = files_and_schema

    clear_database(driver)
    build_supply_chain_graph(driver, products_csv, parts_csv, suppliers_csv, mapping_csv)
    await extract_complaint_graph_from_text(driver, md_path, schema)
    build_the_bridge(driver)
    find_the_culprit(driver)
    
    if driver:
        driver.close()
    print("\n演示完成，已断开数据库连接。")

if __name__ == "__main__":
    asyncio.run(main())