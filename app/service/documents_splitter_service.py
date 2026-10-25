"""
文档切分模块 - 基于 LangChain 的智能文档分割
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.utils.logger import logger
from app.utils.config_handler_util import milvus_conf


class DocumentSplitterService:
    """文档切分服务 基于LangChain构建"""

    def __init__(self):
        """初始化文档切分服务"""
        self.chunk_size = milvus_conf["chunk_size"]
        self.chunk_overlap = milvus_conf["chunk_overlap"]

        # markdown文档分割器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割 避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在切分内容中 保持语义完整
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size 减少分片次数并保持语义完整
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False, # 分隔符参数按字符串解析
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割MarkDown文档 (两阶段分割 + 合并小片段)

        Args:
            content: MarkDown文档内容
            file_path: 文件路径

        Returns:
              List[Document]: 文档分片内容
        """

        if not content or not content.strip():
            logger.warning(f"MarkDown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段 使用文本分割器二次切分
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段 合并太小的分片
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["source"] = file_path
                doc.metadata["extension"] = Path(file_path).suffix.lower() or ".md"
                doc.metadata["file_name"] = Path(file_path).name

            logger.info(f"MarkDown 文档分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise


    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if Path(file_path).suffix.lower() == ".md":
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)


    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径

        Returns:
              List[Document]: 文档分片内容
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "source": file_path,
                        "extension": Path(file_path).suffix.lower(),
                        "file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def _merge_small_chunks(self, docs: List[Document], min_size: int) -> List[Document]:
        """
        合并太小的分片

        Args:
            docs: 分片内容
            min_size: 最小的分片大小

        Returns:
              合并后的分片内容
        """

        if not docs or len(docs) == 0:
            return []

        final_docs = []
        current_doc = None

        for doc in docs:
            if current_doc is None:
                current_doc = doc
            elif (
                    len(doc.page_content) < min_size
                    and len(current_doc.page_content) + len(doc.page_content) <= self.chunk_size * 2
                    # 只在同一章节内合并，避免丢失后续内容的 h1/h2 归属
                    and current_doc.metadata.get("h1") == doc.metadata.get("h1")
                    and current_doc.metadata.get("h2") == doc.metadata.get("h2")
            ):
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                final_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            final_docs.append(current_doc)

        return final_docs

# 全局单例
document_splitter_service = DocumentSplitterService()